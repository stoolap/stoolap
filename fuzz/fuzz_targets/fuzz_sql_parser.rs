#![no_main]

use libfuzzer_sys::fuzz_target;
use stoolap::parser::parse_sql;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, skip invalid UTF-8
    if let Ok(sql) = std::str::from_utf8(data) {
        // The parser should never panic, only return Ok or Err
        let _ = parse_sql(sql);
    }
});
