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

//! Semantic Search Engine — Real-World AI Vector Search Example
//!
//! Demonstrates Stoolap's built-in `EMBED()` function for semantic search:
//!   - Automatic text-to-vector embedding via sentence-transformers (MiniLM-L6-v2)
//!   - HNSW index for fast approximate nearest-neighbor search
//!   - Hybrid search: semantic similarity + SQL filters
//!   - Cross-domain discovery across categories
//!   - Before/after index timing for both end-to-end and ANN-only query paths
//!   - Window functions for per-category ranking
//!
//! The model (~90MB) is automatically downloaded on first run and cached locally.
//!
//! Run:  cargo run --example semantic_search --release --features semantic

use std::collections::BTreeSet;
use std::time::Instant;

use stoolap::api::Database;

// ─────────────────────────────────────────────────────────────
// Document templates — realistic knowledge base articles
// ─────────────────────────────────────────────────────────────

struct Document {
    title: &'static str,
    category: &'static str,
    content: &'static str,
}

const DOCUMENTS: &[Document] = &[
    // ── Technology ──
    Document { title: "Introduction to Machine Learning", category: "Technology", content: "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves." },
    Document { title: "Deep Learning Neural Networks", category: "Technology", content: "Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. Convolutional neural networks excel at image recognition while recurrent networks handle sequential data." },
    Document { title: "Natural Language Processing Advances", category: "Technology", content: "Natural language processing combines computational linguistics with deep learning models to enable computers to understand, interpret, and generate human language. Transformer architectures have revolutionized the field." },
    Document { title: "Cloud Computing Infrastructure", category: "Technology", content: "Cloud computing delivers computing services including servers, storage, databases, networking, and software over the internet. Major providers include AWS, Azure, and Google Cloud Platform." },
    Document { title: "Kubernetes Container Orchestration", category: "Technology", content: "Kubernetes automates the deployment, scaling, and management of containerized applications. It groups containers into logical units for easy management and discovery across clusters of hosts." },
    Document { title: "Blockchain and Distributed Ledger", category: "Technology", content: "Blockchain technology creates a decentralized and distributed digital ledger that records transactions across many computers. This ensures that records cannot be altered retroactively without altering all subsequent blocks." },
    Document { title: "Edge Computing Architecture", category: "Technology", content: "Edge computing brings computation and data storage closer to the sources of data. This reduces latency, saves bandwidth, and enables real-time processing for IoT devices and autonomous systems." },
    Document { title: "Cybersecurity Best Practices", category: "Technology", content: "Cybersecurity involves protecting computer systems, networks, and data from digital attacks. Key practices include encryption, multi-factor authentication, regular security audits, and employee training on phishing threats." },
    Document { title: "Database Management Systems", category: "Technology", content: "Modern database systems range from traditional relational databases using SQL to NoSQL solutions for unstructured data. Embedded databases like SQLite and Stoolap provide serverless data management without requiring a separate server process." },
    Document { title: "Microservices Architecture Patterns", category: "Technology", content: "Microservices architecture structures an application as a collection of loosely coupled services. Each service is fine-grained and implements a single business capability, communicating through lightweight protocols." },
    Document { title: "GraphQL API Design", category: "Technology", content: "GraphQL is a query language for APIs that gives clients the power to ask for exactly what they need. It provides a complete description of the data in your API and makes it easier to evolve APIs over time." },
    Document { title: "WebAssembly Runtime Performance", category: "Technology", content: "WebAssembly enables high-performance applications on the web by providing a binary instruction format for a stack-based virtual machine. It runs at near-native speed and is designed as a portable compilation target." },
    Document { title: "Rust Programming Language Safety", category: "Technology", content: "Rust guarantees memory safety and thread safety at compile time through its ownership system. It achieves C-level performance without garbage collection, making it ideal for systems programming and embedded applications." },
    Document { title: "Vector Databases and Similarity Search", category: "Technology", content: "Vector databases store high-dimensional embeddings and enable fast similarity search using algorithms like HNSW. They power semantic search, recommendation systems, and retrieval-augmented generation in AI applications." },
    Document { title: "Large Language Models Training", category: "Technology", content: "Training large language models requires massive datasets and computational resources. Techniques like attention mechanisms, tokenization, and reinforcement learning from human feedback have driven recent breakthroughs." },

    // ── Science ──
    Document { title: "Quantum Computing Fundamentals", category: "Science", content: "Quantum computing harnesses quantum mechanical phenomena such as superposition and entanglement to process information. Quantum bits or qubits can exist in multiple states simultaneously, enabling exponential computational speedups for certain problems." },
    Document { title: "CRISPR Gene Editing Technology", category: "Science", content: "CRISPR-Cas9 is a revolutionary gene editing technology that allows scientists to modify DNA sequences with unprecedented precision. It has applications in treating genetic diseases, improving crop yields, and understanding gene function." },
    Document { title: "Climate Change Research", category: "Science", content: "Climate science studies the long-term changes in temperature, precipitation, and other atmospheric conditions. Rising greenhouse gas concentrations are causing global warming, leading to sea level rise, extreme weather events, and ecosystem disruption." },
    Document { title: "Neuroscience and Brain Mapping", category: "Science", content: "Modern neuroscience uses advanced imaging techniques like fMRI and optogenetics to map brain connectivity and understand neural circuits. The Human Brain Project aims to create a detailed atlas of the entire human brain." },
    Document { title: "Space Exploration and Mars Missions", category: "Science", content: "Space agencies and private companies are developing technology for Mars exploration and eventual colonization. Challenges include radiation protection, life support systems, and developing propulsion for the long journey." },
    Document { title: "Renewable Energy Technologies", category: "Science", content: "Solar photovoltaic cells and wind turbines are the fastest-growing renewable energy sources. Advances in battery storage technology and grid integration are making renewable energy increasingly cost-competitive with fossil fuels." },
    Document { title: "Particle Physics and the Standard Model", category: "Science", content: "The Standard Model of particle physics describes fundamental particles and their interactions through electromagnetic, weak, and strong nuclear forces. The discovery of the Higgs boson confirmed a key prediction of this framework." },
    Document { title: "Synthetic Biology Applications", category: "Science", content: "Synthetic biology combines engineering principles with biology to design and construct new biological parts, devices, and systems. Applications range from biofuel production to creating organisms that can detect environmental pollutants." },
    Document { title: "Ocean Acidification Effects", category: "Science", content: "Ocean acidification occurs when seawater absorbs excess carbon dioxide from the atmosphere, lowering its pH. This threatens marine ecosystems, particularly coral reefs and shellfish, which depend on calcium carbonate for their structures." },
    Document { title: "Gravitational Waves Detection", category: "Science", content: "Gravitational waves are ripples in spacetime caused by accelerating massive objects. LIGO and Virgo observatories detect these waves from events like black hole mergers, opening a new window into astrophysics." },

    // ── Healthcare ──
    Document { title: "AI in Medical Diagnostics", category: "Healthcare", content: "Artificial intelligence algorithms can analyze medical images, pathology slides, and patient records to assist in diagnosis. Deep learning models have achieved physician-level accuracy in detecting certain cancers, eye diseases, and cardiac conditions." },
    Document { title: "Telemedicine and Remote Care", category: "Healthcare", content: "Telemedicine enables patients to consult with healthcare providers remotely through video calls, messaging, and remote monitoring devices. The technology has expanded access to healthcare in rural areas and reduced unnecessary emergency visits." },
    Document { title: "Personalized Medicine and Genomics", category: "Healthcare", content: "Personalized medicine uses an individual's genetic profile to guide decisions about disease prevention, diagnosis, and treatment. Pharmacogenomics helps determine which medications and dosages will be most effective for each patient." },
    Document { title: "Mental Health Treatment Innovation", category: "Healthcare", content: "New approaches to mental health treatment include digital therapeutics, ketamine-assisted therapy, and AI-powered chatbots for cognitive behavioral therapy. These innovations aim to address the growing global mental health crisis." },
    Document { title: "Vaccine Development Process", category: "Healthcare", content: "Modern vaccine development uses mRNA technology, viral vectors, and protein subunit approaches. The rapid development of COVID-19 vaccines demonstrated how new platforms can accelerate the traditional multi-year development timeline." },
    Document { title: "Wearable Health Monitoring", category: "Healthcare", content: "Wearable devices can continuously monitor heart rate, blood oxygen, sleep patterns, and physical activity. Advanced sensors are being developed to track blood glucose levels, detect atrial fibrillation, and predict health events." },
    Document { title: "Robotic Surgery Advances", category: "Healthcare", content: "Robotic surgical systems provide surgeons with enhanced precision, flexibility, and control during minimally invasive procedures. The technology reduces patient recovery time, blood loss, and post-operative complications." },
    Document { title: "Drug Discovery with AI", category: "Healthcare", content: "Artificial intelligence accelerates drug discovery by predicting molecular properties, identifying potential drug targets, and optimizing clinical trial design. Machine learning models can screen millions of compounds in days rather than years." },
    Document { title: "Epidemiology and Disease Tracking", category: "Healthcare", content: "Epidemiologists use statistical methods, genomic sequencing, and digital surveillance to track disease outbreaks and predict pandemic spread. Wastewater monitoring has emerged as an early warning system for community infection levels." },
    Document { title: "Elderly Care Technology", category: "Healthcare", content: "Technology solutions for elderly care include fall detection systems, medication management apps, social robots for companionship, and smart home sensors that monitor daily activities and alert caregivers to changes in routine." },

    // ── Business ──
    Document { title: "Startup Funding and Venture Capital", category: "Business", content: "Venture capital firms invest in high-growth startups in exchange for equity. The funding lifecycle typically progresses through seed, Series A, B, and C rounds, with each stage requiring demonstrated growth metrics and market validation." },
    Document { title: "Remote Work Productivity", category: "Business", content: "Remote work has transformed business operations, requiring new management approaches, collaboration tools, and productivity metrics. Companies are adopting hybrid models that combine the flexibility of remote work with in-person collaboration." },
    Document { title: "Supply Chain Optimization", category: "Business", content: "Modern supply chain management uses AI, IoT sensors, and blockchain for real-time visibility, demand forecasting, and inventory optimization. Resilient supply chains diversify suppliers and maintain strategic inventory buffers." },
    Document { title: "Digital Marketing Analytics", category: "Business", content: "Digital marketing leverages data analytics to measure campaign performance, customer engagement, and return on investment. Tools like A/B testing, attribution modeling, and customer journey mapping help optimize marketing strategies." },
    Document { title: "Corporate ESG Reporting", category: "Business", content: "Environmental, social, and governance reporting has become essential for corporate transparency. Investors increasingly evaluate companies based on carbon emissions, diversity metrics, labor practices, and board governance structures." },
    Document { title: "Mergers and Acquisitions Strategy", category: "Business", content: "Mergers and acquisitions involve complex due diligence, valuation, and integration processes. Successful M&A strategies focus on cultural alignment, synergy realization, and retaining key talent during the transition period." },
    Document { title: "E-commerce Platform Development", category: "Business", content: "E-commerce platforms must balance user experience, payment security, inventory management, and logistics. Headless commerce architectures separate the frontend presentation from backend services for greater flexibility." },
    Document { title: "Financial Risk Management", category: "Business", content: "Financial risk management identifies, assesses, and mitigates potential losses from market volatility, credit defaults, and operational failures. Stress testing and scenario analysis help organizations prepare for adverse conditions." },
    Document { title: "Customer Retention Strategies", category: "Business", content: "Customer retention programs use loyalty rewards, personalized communications, and proactive support to reduce churn. Data analytics helps identify at-risk customers and the factors that drive long-term customer satisfaction." },
    Document { title: "Agile Project Management", category: "Business", content: "Agile methodology emphasizes iterative development, cross-functional teams, and continuous feedback. Scrum and Kanban frameworks help teams deliver value incrementally while adapting to changing requirements and priorities." },

    // ── Education ──
    Document { title: "Online Learning Platforms", category: "Education", content: "Online learning platforms provide accessible education through video lectures, interactive exercises, and peer discussion forums. Adaptive learning algorithms personalize content delivery based on individual student progress and learning patterns." },
    Document { title: "AI Tutoring Systems", category: "Education", content: "AI-powered tutoring systems provide personalized instruction by adapting to each student's knowledge level and learning style. These systems use natural language processing to understand student questions and generate targeted explanations." },
    Document { title: "STEM Education Initiatives", category: "Education", content: "STEM education programs emphasize science, technology, engineering, and mathematics skills through hands-on projects and real-world applications. Coding bootcamps and maker spaces provide alternative pathways to technology careers." },
    Document { title: "Educational Assessment Methods", category: "Education", content: "Modern assessment methods go beyond traditional testing to include portfolio evaluation, project-based assessment, and competency demonstrations. Formative assessment provides ongoing feedback to guide learning rather than just measure outcomes." },
    Document { title: "Lifelong Learning and Reskilling", category: "Education", content: "The rapid pace of technological change requires continuous learning and skill development throughout careers. Micro-credentials, professional certificates, and corporate training programs help workers adapt to evolving job requirements." },
    Document { title: "Special Education Technology", category: "Education", content: "Assistive technology helps students with disabilities access educational content through screen readers, speech-to-text tools, and adaptive interfaces. Universal design principles create learning environments that accommodate diverse needs." },
    Document { title: "Early Childhood Development", category: "Education", content: "Early childhood education research shows that quality learning experiences in the first five years significantly impact cognitive development, social skills, and long-term academic success. Play-based learning is particularly effective for young children." },
    Document { title: "University Research Funding", category: "Education", content: "University research relies on federal grants, industry partnerships, and philanthropic donations. Funding agencies evaluate proposals based on scientific merit, broader impacts, and the potential for transformative discoveries." },

    // ── Environment ──
    Document { title: "Carbon Capture Technology", category: "Environment", content: "Carbon capture and storage technology removes carbon dioxide from industrial emissions or directly from the atmosphere. Methods include chemical absorption, membrane separation, and direct air capture with geological storage." },
    Document { title: "Biodiversity Conservation", category: "Environment", content: "Biodiversity conservation efforts protect endangered species and their habitats through wildlife corridors, protected areas, and breeding programs. Citizen science and DNA barcoding help monitor species populations across ecosystems." },
    Document { title: "Sustainable Agriculture Practices", category: "Environment", content: "Sustainable agriculture reduces environmental impact through crop rotation, organic farming, precision irrigation, and integrated pest management. Regenerative farming practices rebuild soil health while maintaining productive yields." },
    Document { title: "Plastic Pollution Solutions", category: "Environment", content: "Addressing plastic pollution requires reducing single-use plastics, improving recycling infrastructure, and developing biodegradable alternatives. Ocean cleanup technologies and extended producer responsibility programs complement prevention efforts." },
    Document { title: "Water Resource Management", category: "Environment", content: "Water resource management balances human consumption, agricultural irrigation, and ecosystem needs. Desalination, water recycling, and smart irrigation systems help address growing water scarcity in drought-prone regions." },
    Document { title: "Urban Green Infrastructure", category: "Environment", content: "Green infrastructure in cities includes parks, green roofs, rain gardens, and urban forests that manage stormwater, reduce heat islands, and improve air quality. These nature-based solutions provide multiple environmental and social benefits." },
    Document { title: "Deforestation and Reforestation", category: "Environment", content: "Tropical deforestation releases stored carbon and destroys wildlife habitats. Reforestation and afforestation programs plant billions of trees annually, but success depends on choosing native species and protecting newly planted areas." },
    Document { title: "Electric Vehicle Transition", category: "Environment", content: "The transition to electric vehicles reduces transportation emissions and air pollution. Battery technology improvements are extending range, reducing costs, and enabling vehicle-to-grid energy storage that supports renewable energy integration." },

    // ── Finance ──
    Document { title: "Cryptocurrency Market Dynamics", category: "Finance", content: "Cryptocurrency markets operate 24/7 with high volatility driven by speculation, regulatory news, and technological developments. Bitcoin and Ethereum remain dominant, while decentralized finance protocols create new financial instruments." },
    Document { title: "Algorithmic Trading Strategies", category: "Finance", content: "Algorithmic trading uses mathematical models and high-speed computing to execute trades automatically. Strategies include market making, statistical arbitrage, and momentum trading, with machine learning increasingly driving signal generation." },
    Document { title: "Central Bank Digital Currencies", category: "Finance", content: "Central banks worldwide are exploring digital currencies that would provide the convenience of digital payments with the stability of government-backed money. CBDCs could improve financial inclusion and reduce transaction costs." },
    Document { title: "Insurance Technology Innovation", category: "Finance", content: "Insurtech companies use AI, telematics, and big data analytics to personalize insurance pricing, automate claims processing, and detect fraud. Usage-based insurance and parametric products are transforming traditional insurance models." },
    Document { title: "Personal Financial Planning", category: "Finance", content: "Personal financial planning involves budgeting, saving, investing, and retirement planning. Robo-advisors use algorithms to provide automated investment management at lower costs than traditional financial advisors." },
    Document { title: "Sustainable Investing and ESG Funds", category: "Finance", content: "Sustainable investing integrates environmental, social, and governance factors into investment decisions. ESG-focused funds have grown significantly as investors seek both financial returns and positive societal impact." },

    // ── Law and Policy ──
    Document { title: "Data Privacy Regulations", category: "Law", content: "Data privacy laws like GDPR and CCPA give individuals control over their personal data. Organizations must implement data protection measures, obtain consent for data processing, and provide mechanisms for data access and deletion requests." },
    Document { title: "AI Ethics and Regulation", category: "Law", content: "AI regulation addresses algorithmic bias, transparency, accountability, and safety. The EU AI Act classifies AI systems by risk level and imposes requirements ranging from transparency obligations to outright bans on certain uses." },
    Document { title: "Intellectual Property in the Digital Age", category: "Law", content: "Digital technology creates new challenges for intellectual property protection. Issues include software patents, copyright for AI-generated content, open source licensing, and the balance between innovation incentives and public access." },
    Document { title: "International Trade Agreements", category: "Law", content: "Trade agreements establish rules for tariffs, market access, and dispute resolution between countries. Modern agreements increasingly address digital trade, environmental standards, and labor protections alongside traditional goods and services." },
    Document { title: "Antitrust and Big Tech Regulation", category: "Law", content: "Antitrust regulators scrutinize large technology companies for anti-competitive practices including self-preferencing, acquisition of potential competitors, and leveraging platform dominance across markets." },
    Document { title: "Immigration Policy and Workforce", category: "Law", content: "Immigration policies affect the global technology workforce through visa programs, skill-based immigration systems, and international talent mobility. Countries compete to attract highly skilled workers in engineering and science fields." },

    // ── Arts and Culture ──
    Document { title: "AI-Generated Art Controversy", category: "Culture", content: "AI image generators like DALL-E and Midjourney create artwork from text descriptions, raising questions about creativity, copyright, and the future of artistic professions. The technology democratizes art creation while challenging traditional notions of authorship." },
    Document { title: "Streaming Platform Impact on Music", category: "Culture", content: "Music streaming platforms have transformed how artists distribute and monetize their work. While increasing access to global audiences, the economics of streaming have reduced per-play revenue, favoring artists with high stream counts." },
    Document { title: "Virtual Reality in Entertainment", category: "Culture", content: "Virtual reality technology creates immersive entertainment experiences in gaming, film, and live events. Social VR platforms enable shared experiences, while volumetric video capture brings real performers into virtual environments." },
    Document { title: "Digital Preservation of Heritage", category: "Culture", content: "Digital technology preserves cultural heritage through 3D scanning of artifacts, virtual museum tours, and digitization of historical documents. These efforts protect irreplaceable cultural assets from natural disasters and conflict." },
    Document { title: "Podcasting and Audio Content Growth", category: "Culture", content: "Podcasting has grown into a major media format with millions of shows covering every topic. The medium's intimate format and on-demand nature have created new opportunities for storytelling, education, and community building." },

    // ── Social Science ──
    Document { title: "Social Media and Mental Health", category: "Social Science", content: "Research examines the relationship between social media use and mental health outcomes, particularly among adolescents. Studies indicate correlations between excessive screen time and anxiety, depression, and body image concerns." },
    Document { title: "Urban Planning and Smart Cities", category: "Social Science", content: "Smart city initiatives use IoT sensors, data analytics, and digital infrastructure to improve urban services. Applications include intelligent traffic management, energy-efficient buildings, and predictive maintenance of public infrastructure." },
    Document { title: "Demographic Trends and Aging Population", category: "Social Science", content: "Many developed countries face aging populations and declining birth rates. These demographic shifts affect healthcare systems, pension programs, labor markets, and economic growth, requiring policy adaptations and technological solutions." },
    Document { title: "Digital Divide and Internet Access", category: "Social Science", content: "The digital divide refers to unequal access to internet connectivity and digital literacy. Satellite internet, community networks, and digital skills programs aim to bridge the gap between connected and underserved communities." },
    Document { title: "Behavioral Economics Insights", category: "Social Science", content: "Behavioral economics studies how psychological factors influence economic decisions. Concepts like loss aversion, anchoring, and choice architecture inform policy design, marketing strategies, and personal financial decision-making." },
];

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

fn print_separator(width: usize) {
    println!("{}", "=".repeat(width));
}

fn category_count() -> usize {
    DOCUMENTS
        .iter()
        .map(|d| d.category)
        .collect::<BTreeSet<_>>()
        .len()
}

fn docs_per_second(total_docs: usize, elapsed_secs: f64) -> f64 {
    if elapsed_secs > f64::EPSILON {
        total_docs as f64 / elapsed_secs
    } else {
        total_docs as f64
    }
}

fn avg_query_time_ms(db: &Database, sql: &str, runs: usize) -> f64 {
    let runs = runs.max(1);
    let mut total_ms = 0.0;
    for _ in 0..runs {
        let start = Instant::now();
        for row in db.query(sql, ()).expect("query failed") {
            row.expect("row error");
        }
        total_ms += start.elapsed().as_secs_f64() * 1000.0;
    }
    total_ms / runs as f64
}

fn run_query_and_print(db: &Database, label: &str, sql: &str, columns: &[(&str, usize, &str)]) {
    println!();
    println!("  {label}");
    println!("  {}", "-".repeat(70));
    println!();

    let start = Instant::now();
    let mut count = 0;

    // Print header
    print!("  ");
    for (name, width, _) in columns {
        print!("{:>width$}", name, width = width);
    }
    println!();
    print!("  ");
    for (_, width, _) in columns {
        print!("{:>width$}", "-".repeat(*width), width = width);
    }
    println!();

    for row in db.query(sql, ()).expect("query failed") {
        let row = row.expect("row error");
        print!("  ");
        for (col_idx, (_, width, fmt)) in columns.iter().enumerate() {
            match *fmt {
                "text" => {
                    let val = row.get::<String>(col_idx).unwrap_or_default();
                    let truncated = if val.len() > width - 2 {
                        format!("{}...", &val[..width - 5])
                    } else {
                        val
                    };
                    print!("{:<width$}", truncated, width = width);
                }
                "f4" => {
                    let val = row.get::<f64>(col_idx).unwrap_or(0.0);
                    print!("{:>width$.4}", val, width = width);
                }
                "int" => {
                    let val = row.get::<i64>(col_idx).unwrap_or(0);
                    print!("{:>width$}", val, width = width);
                }
                _ => {}
            }
        }
        println!();
        count += 1;
    }

    let elapsed = start.elapsed();
    println!();
    println!(
        "  ({} rows, {:.1}ms)",
        count,
        elapsed.as_secs_f64() * 1000.0
    );
}

fn main() {
    println!();
    print_separator(76);
    println!("  STOOLAP SEMANTIC SEARCH ENGINE");
    println!("  Built-in AI Embeddings — No External APIs Required");
    print_separator(76);
    println!();
    println!("  Model:  sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)");
    println!(
        "  Docs:   {} articles across {} categories",
        DOCUMENTS.len(),
        category_count()
    );
    println!();

    // ── Create database ──
    let db = Database::open_in_memory().expect("failed to open database");

    db.execute(
        "CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384)
        )",
        (),
    )
    .unwrap();

    // ── Insert documents with EMBED() ──
    print!("  Generating embeddings and inserting documents...");
    let insert_start = Instant::now();

    db.execute("BEGIN", ()).unwrap();
    for (i, doc) in DOCUMENTS.iter().enumerate() {
        let title_escaped = doc.title.replace('\'', "''");
        let content_escaped = doc.content.replace('\'', "''");
        let embed_text = format!("{}. {}", doc.title, doc.content).replace('\'', "''");
        db.execute(
            &format!(
                "INSERT INTO documents (id, title, category, content, embedding) \
                 VALUES ({i}, '{title_escaped}', '{}', '{content_escaped}', EMBED('{embed_text}'))",
                doc.category
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    let insert_time = insert_start.elapsed();
    let insert_secs = insert_time.as_secs_f64();
    let insert_rate = docs_per_second(DOCUMENTS.len(), insert_secs);
    println!(" done ({:.1}s, {:.1} docs/s)", insert_secs, insert_rate);

    // ── Benchmark top-k query before index (after warm-up) ──
    let benchmark_sql_end_to_end =
        "WITH query AS (SELECT EMBED('How does artificial intelligence work?') AS vec) \
         SELECT id, VEC_DISTANCE_COSINE(embedding, query.vec) AS dist \
         FROM documents, query \
         ORDER BY dist LIMIT 10";
    db.execute("CREATE TABLE query_vec (vec VECTOR(384))", ())
        .unwrap();
    db.execute(
        "INSERT INTO query_vec(vec) VALUES (EMBED('How does artificial intelligence work?'))",
        (),
    )
    .unwrap();
    let benchmark_sql_ann_only =
        "SELECT id, VEC_DISTANCE_COSINE(embedding, query_vec.vec) AS dist \
         FROM documents, query_vec \
         ORDER BY dist LIMIT 10";
    let benchmark_runs = 10usize;
    let _ = avg_query_time_ms(&db, benchmark_sql_end_to_end, 1); // warm-up to avoid first-load model cost
    let _ = avg_query_time_ms(&db, benchmark_sql_ann_only, 1);
    let baseline_e2e_ms = avg_query_time_ms(&db, benchmark_sql_end_to_end, benchmark_runs);
    let baseline_ann_ms = avg_query_time_ms(&db, benchmark_sql_ann_only, benchmark_runs);
    println!(
        "  Baseline top-k (EMBED + search), no index: {:.2}ms avg over {} runs",
        baseline_e2e_ms, benchmark_runs
    );
    println!(
        "  Baseline top-k (ANN only, precomputed vec), no index: {:.2}ms avg over {} runs",
        baseline_ann_ms, benchmark_runs
    );

    // ── Build HNSW index ──
    print!("  Building HNSW cosine index...");
    let build_start = Instant::now();
    db.execute(
        "CREATE INDEX idx_doc_embedding ON documents(embedding) USING HNSW WITH (metric = 'cosine')",
        (),
    )
    .unwrap();
    let build_time = build_start.elapsed();
    println!(" done ({:.2}s)", build_time.as_secs_f64());
    let indexed_e2e_ms = avg_query_time_ms(&db, benchmark_sql_end_to_end, benchmark_runs);
    let indexed_ann_ms = avg_query_time_ms(&db, benchmark_sql_ann_only, benchmark_runs);
    println!(
        "  Same top-k (EMBED + search) with HNSW: {:.2}ms avg over {} runs",
        indexed_e2e_ms, benchmark_runs
    );
    println!(
        "  Same top-k (ANN only, precomputed vec) with HNSW: {:.2}ms avg over {} runs",
        indexed_ann_ms, benchmark_runs
    );
    if baseline_e2e_ms > 0.0 && indexed_e2e_ms > 0.0 {
        let ratio = baseline_e2e_ms / indexed_e2e_ms;
        if ratio >= 1.0 {
            println!("  End-to-end speedup: {:.2}x", ratio);
        } else {
            println!("  End-to-end slowdown: {:.2}x", 1.0 / ratio);
        }
    }
    if baseline_ann_ms > 0.0 && indexed_ann_ms > 0.0 {
        let ratio = baseline_ann_ms / indexed_ann_ms;
        if ratio >= 1.0 {
            println!("  ANN-only speedup: {:.2}x", ratio);
        } else {
            println!("  ANN-only slowdown: {:.2}x", 1.0 / ratio);
        }
    }
    println!(
        "  Note: end-to-end timings include EMBED() inference; ANN-only isolates vector index impact."
    );

    // ── Create supporting indexes ──
    db.execute("CREATE INDEX idx_doc_category ON documents(category)", ())
        .unwrap();

    println!();
    print_separator(76);
    println!("  SEMANTIC SEARCH RESULTS");
    print_separator(76);

    // ══════════════════════════════════════════════════════════
    // Scenario 1: Basic semantic search
    // ══════════════════════════════════════════════════════════

    run_query_and_print(
        &db,
        "[1] Semantic Search: 'How does artificial intelligence work?'",
        "WITH query AS (SELECT EMBED('How does artificial intelligence work?') AS vec) \
         SELECT title, category, \
                VEC_DISTANCE_COSINE(embedding, query.vec) AS dist \
         FROM documents, query \
         ORDER BY dist LIMIT 10",
        &[
            ("Title", 45, "text"),
            ("Category", 15, "text"),
            ("Distance", 12, "f4"),
        ],
    );

    // ══════════════════════════════════════════════════════════
    // Scenario 2: Cross-domain discovery
    // ══════════════════════════════════════════════════════════

    run_query_and_print(
        &db,
        "[2] Cross-Domain: 'environmental sustainability and green technology'",
        "WITH query AS (SELECT EMBED('environmental sustainability and green technology') AS vec) \
         SELECT title, category, \
                VEC_DISTANCE_COSINE(embedding, query.vec) AS dist \
         FROM documents, query \
         ORDER BY dist LIMIT 10",
        &[
            ("Title", 45, "text"),
            ("Category", 15, "text"),
            ("Distance", 12, "f4"),
        ],
    );

    // ══════════════════════════════════════════════════════════
    // Scenario 3: Hybrid search (category filter + semantic)
    // ══════════════════════════════════════════════════════════

    run_query_and_print(
        &db,
        "[3] Hybrid: Healthcare articles about 'using technology to help patients'",
        "WITH query AS (SELECT EMBED('using technology to help patients') AS vec) \
         SELECT title, \
                VEC_DISTANCE_COSINE(embedding, query.vec) AS dist \
         FROM documents, query \
         WHERE category = 'Healthcare' \
         ORDER BY dist LIMIT 5",
        &[("Title", 55, "text"), ("Distance", 12, "f4")],
    );

    // ══════════════════════════════════════════════════════════
    // Scenario 4: Concept search (no keyword match)
    // ══════════════════════════════════════════════════════════

    run_query_and_print(
        &db,
        "[4] Concept Search: 'protecting personal information online' (no keyword overlap)",
        "WITH query AS (SELECT EMBED('protecting personal information online') AS vec) \
         SELECT title, category, \
                VEC_DISTANCE_COSINE(embedding, query.vec) AS dist \
         FROM documents, query \
         ORDER BY dist LIMIT 8",
        &[
            ("Title", 45, "text"),
            ("Category", 15, "text"),
            ("Distance", 12, "f4"),
        ],
    );

    // ══════════════════════════════════════════════════════════
    // Scenario 5: Best match per category (window function)
    // ══════════════════════════════════════════════════════════

    run_query_and_print(
        &db,
        "[5] Best Match Per Category: 'future of work and automation'",
        "WITH query AS (SELECT EMBED('future of work and automation') AS vec) \
         SELECT title, category, dist FROM ( \
            SELECT title, category, \
                   VEC_DISTANCE_COSINE(embedding, query.vec) AS dist, \
                   RANK() OVER (PARTITION BY category ORDER BY VEC_DISTANCE_COSINE(embedding, query.vec)) AS rnk \
            FROM documents, query \
         ) sub WHERE rnk = 1 ORDER BY dist",
        &[("Title", 45, "text"), ("Category", 15, "text"), ("Distance", 12, "f4")],
    );

    // ══════════════════════════════════════════════════════════
    // Scenario 6: Aggregation — category overview
    // ══════════════════════════════════════════════════════════

    run_query_and_print(
        &db,
        "[6] Knowledge Base Overview",
        "SELECT category, COUNT(*) AS docs \
         FROM documents \
         GROUP BY category \
         ORDER BY docs DESC",
        &[("Category", 20, "text"), ("Documents", 12, "int")],
    );

    // ══════════════════════════════════════════════════════════
    // Scenario 7: Question answering style
    // ══════════════════════════════════════════════════════════

    run_query_and_print(
        &db,
        "[7] Question: 'What are the risks of social media for teenagers?'",
        "WITH query AS (SELECT EMBED('What are the risks of social media for teenagers?') AS vec) \
         SELECT title, category, content, \
                VEC_DISTANCE_COSINE(embedding, query.vec) AS dist \
         FROM documents, query \
         ORDER BY dist LIMIT 3",
        &[
            ("Title", 40, "text"),
            ("Category", 14, "text"),
            ("Content", 50, "text"),
            ("Dist", 10, "f4"),
        ],
    );

    println!();
    print_separator(76);
    println!("  PERFORMANCE SUMMARY");
    print_separator(76);
    println!();
    println!("    Documents:     {}", DOCUMENTS.len());
    println!(
        "    Embedding:     {:.1}s ({:.1} docs/s)",
        insert_secs, insert_rate
    );
    println!("    Index build:   {:.2}s", build_time.as_secs_f64());
    println!("    Model:         all-MiniLM-L6-v2 (384 dims)");
    println!("    Distance:      Cosine similarity via HNSW index");
    println!();
    println!("  No external APIs. No Python. No Docker. Just SQL.");
    println!();
    print_separator(76);
    println!();
}
