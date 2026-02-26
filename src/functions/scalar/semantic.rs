// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Semantic embedding function — EMBED(text) → VECTOR(384)
//!
//! Uses the `sentence-transformers/all-MiniLM-L6-v2` model via Hugging Face's
//! candle framework (pure Rust) to convert text into 384-dimensional embeddings.
//!
//! The model is lazily downloaded from Hugging Face Hub on first call and cached
//! in `~/.cache/huggingface/hub/`. Subsequent calls reuse the cached model.
//!
//! # Example
//!
//! ```sql
//! -- Generate an embedding from text
//! SELECT EMBED('How to reset my password');
//!
//! -- Insert with auto-generated embedding
//! INSERT INTO docs (content, embedding) VALUES ('Hello world', EMBED('Hello world'));
//!
//! -- Semantic search
//! SELECT content FROM docs
//! ORDER BY VEC_DISTANCE_COSINE(embedding, EMBED('greeting')) LIMIT 5;
//! ```

use std::sync::OnceLock;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

use crate::core::{Error, Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};

// ============================================================================
// Global model singleton
// ============================================================================

struct EmbedModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    hidden_size: usize,
}

static EMBED_MODEL: OnceLock<std::result::Result<parking_lot::RwLock<EmbedModel>, String>> =
    OnceLock::new();

const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Initialize the embedding model (called once, lazily).
fn init_model() -> std::result::Result<EmbedModel, String> {
    let api = hf_hub::api::sync::Api::new().map_err(|e| format!("HF Hub init failed: {e}"))?;
    let repo = api.model(DEFAULT_MODEL.to_string());

    let config_path = repo
        .get("config.json")
        .map_err(|e| format!("Failed to download config.json: {e}"))?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| format!("Failed to download tokenizer.json: {e}"))?;
    let weights_path = repo
        .get("model.safetensors")
        .map_err(|e| format!("Failed to download model.safetensors: {e}"))?;

    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read config.json: {e}"))?;
    let config: Config =
        serde_json::from_str(&config_str).map_err(|e| format!("Failed to parse config: {e}"))?;

    let hidden_size = config.hidden_size;
    let device = Device::Cpu;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
            .map_err(|e| format!("Failed to load weights: {e}"))?
    };

    let model =
        BertModel::load(vb, &config).map_err(|e| format!("Failed to build BERT model: {e}"))?;

    Ok(EmbedModel {
        model,
        tokenizer,
        device,
        hidden_size,
    })
}

/// Get or initialize the global embedding model.
fn get_model() -> Result<&'static parking_lot::RwLock<EmbedModel>> {
    let result = EMBED_MODEL.get_or_init(|| init_model().map(parking_lot::RwLock::new));
    match result {
        Ok(m) => Ok(m),
        Err(e) => Err(Error::Internal {
            message: format!("EMBED model initialization failed: {e}"),
        }),
    }
}

// ============================================================================
// Core embedding logic
// ============================================================================

/// Embed a single text string into a f32 vector.
fn embed_text(text: &str) -> Result<Vec<f32>> {
    let model_lock = get_model()?;
    let model_guard = model_lock.read();

    let encoding = model_guard
        .tokenizer
        .encode(text, true)
        .map_err(|e| Error::internal(format!("Tokenization failed: {e}")))?;

    let token_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    let seq_len = token_ids.len();
    let device = &model_guard.device;

    // Build tensors [1, seq_len]
    let token_ids_tensor = Tensor::new(token_ids, device)
        .and_then(|t| t.reshape((1, seq_len)))
        .map_err(|e| Error::internal(format!("Token tensor error: {e}")))?;

    let type_ids_tensor = Tensor::zeros((1, seq_len), DType::U32, device)
        .map_err(|e| Error::internal(format!("Type IDs tensor error: {e}")))?;

    let attention_mask_u32: Vec<u32> = attention_mask.to_vec();
    let attention_mask_tensor = Tensor::new(attention_mask_u32.as_slice(), device)
        .and_then(|t| t.reshape((1, seq_len)))
        .map_err(|e| Error::internal(format!("Attention mask tensor error: {e}")))?;

    // Forward pass → [1, seq_len, hidden_size]
    let embeddings = model_guard
        .model
        .forward(
            &token_ids_tensor,
            &type_ids_tensor,
            Some(&attention_mask_tensor),
        )
        .map_err(|e| Error::internal(format!("Model forward pass failed: {e}")))?;

    // Mean pooling with attention mask
    let hidden_size = model_guard.hidden_size;
    let attention_f32 = attention_mask_tensor
        .to_dtype(DType::F32)
        .map_err(|e| Error::internal(format!("Dtype conversion error: {e}")))?;

    // [1, seq_len, 1] for broadcasting
    let mask_3d = attention_f32
        .reshape((1, seq_len, 1))
        .map_err(|e| Error::internal(format!("Mask reshape error: {e}")))?;

    // Mask padding tokens: [1, seq_len, hidden_size]
    let masked = embeddings
        .broadcast_mul(&mask_3d)
        .map_err(|e| Error::internal(format!("Broadcast mul error: {e}")))?;

    // Sum over seq_len → [1, hidden_size]
    let sum = masked
        .sum(1)
        .map_err(|e| Error::internal(format!("Sum error: {e}")))?;

    // Count real tokens → [1, 1]
    let count = attention_f32
        .sum(1)
        .and_then(|t| t.reshape((1, 1)))
        .map_err(|e| Error::internal(format!("Count error: {e}")))?;

    // Mean → [1, hidden_size]
    let pooled = sum
        .broadcast_div(&count)
        .map_err(|e| Error::internal(format!("Div error: {e}")))?;

    // L2-normalize
    let pooled_sq = pooled
        .sqr()
        .and_then(|t| t.sum(1))
        .and_then(|t| t.sqrt())
        .and_then(|t| t.reshape((1, 1)))
        .map_err(|e| Error::internal(format!("Norm error: {e}")))?;

    let normalized = pooled
        .broadcast_div(&pooled_sq)
        .map_err(|e| Error::internal(format!("Normalize error: {e}")))?;

    // Extract as Vec<f32> — [1, hidden_size] → flatten
    let result: Vec<f32> = normalized
        .reshape(hidden_size)
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| Error::internal(format!("Result extraction error: {e}")))?;

    Ok(result)
}

// ============================================================================
// SQL function: EMBED(text) → VECTOR(384)
// ============================================================================

/// EMBED(text) — Convert text to a 384-dimensional semantic embedding vector.
///
/// Uses the `sentence-transformers/all-MiniLM-L6-v2` model (pure Rust inference).
/// The model is automatically downloaded on first use and cached locally.
#[derive(Default)]
pub struct EmbedFunction;

impl ScalarFunction for EmbedFunction {
    fn name(&self) -> &str {
        "EMBED"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "EMBED",
            FunctionType::Scalar,
            "Convert text to a 384-dim semantic embedding vector (MiniLM-L6-v2)",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(EmbedFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        crate::validate_arg_count!(args, "EMBED", 1);

        let text = match &args[0] {
            Value::Null(_) => return Ok(Value::null_unknown()),
            Value::Text(s) => s.as_ref(),
            Value::Integer(i) => return embed_and_wrap(&i.to_string()),
            Value::Float(f) => return embed_and_wrap(&f.to_string()),
            other => {
                return Err(Error::invalid_argument(format!(
                    "EMBED requires TEXT argument, got {:?}",
                    other.data_type()
                )));
            }
        };

        embed_and_wrap(text)
    }
}

/// Embed text and wrap as Value::vector
fn embed_and_wrap(text: &str) -> Result<Value> {
    let embedding = embed_text(text)?;
    Ok(Value::vector(embedding))
}
