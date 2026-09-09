extern crate roaring;
use roaring::RoaringTreemap;

#[test]
fn smoke() {
    let mut bitmap = RoaringTreemap::new();
    assert_eq!(bitmap.len(), 0);
    assert!(bitmap.is_empty());
    bitmap.remove(0);
    assert_eq!(bitmap.len(), 0);
    assert!(bitmap.is_empty());
    bitmap.insert(1);
    assert!(bitmap.contains(1));
    assert_eq!(bitmap.len(), 1);
    assert!(!bitmap.is_empty());
    bitmap.insert(u64::MAX - 2);
    assert!(bitmap.contains(u64::MAX - 2));
    assert_eq!(bitmap.len(), 2);
    bitmap.insert(u64::MAX);
    assert!(bitmap.contains(u64::MAX));
    assert_eq!(bitmap.len(), 3);
    bitmap.insert(2);
    assert!(bitmap.contains(2));
    assert_eq!(bitmap.len(), 4);
    bitmap.remove(2);
    assert!(!bitmap.contains(2));
    assert_eq!(bitmap.len(), 3);
    assert!(!bitmap.contains(0));
    assert!(bitmap.contains(1));
    assert!(!bitmap.contains(100));
    assert!(bitmap.contains(u64::MAX - 2));
    assert!(!bitmap.contains(u64::MAX - 1));
    assert!(bitmap.contains(u64::MAX));
}

#[test]
fn insert_range() {
    let ranges = 0..0x1000;
    const SIGMA: u64 = u32::MAX as u64;

    let mut bitmap = RoaringTreemap::new();
    assert_eq!(bitmap.insert_range(ranges), 0x1000);
    assert_eq!(bitmap.len(), 0x1000);
    assert_eq!(bitmap.max(), Some(0xFFF));

    assert_eq!(bitmap.insert_range(u32::MAX as u64 - 1..u32::MAX as u64 + 1), 2);
    assert!(bitmap.contains(2));
    assert!(bitmap.contains(0xFFF));
    assert!(!bitmap.contains(0x1000));

    bitmap.clear();
    bitmap.insert_range(2 * SIGMA..=4 * SIGMA);

    assert_eq!(bitmap.min(), Some(2 * SIGMA));
    assert_eq!(bitmap.max(), Some(4 * SIGMA));

    assert!(bitmap.contains(3 * SIGMA));
}

#[test]
fn remove_range() {
    let ranges = [0u64, 1, 63, 64, 65, 100, 4096 - 1, 4096, 4096 + 1, 65536 - 1];
    for (i, &a) in ranges.iter().enumerate() {
        for &b in &ranges[i..] {
            let mut bitmap = (0..=65536).collect::<RoaringTreemap>();
            assert_eq!(bitmap.remove_range(a..b), (b - a));
            assert_eq!(bitmap, ((0..a).chain(b..=65536)).collect::<RoaringTreemap>());
        }
    }
}

#[test]
fn test_max() {
    let mut bitmap = RoaringTreemap::new();
    assert_eq!(bitmap.max(), None);
    bitmap.insert(0);
    assert_eq!(bitmap.max(), Some(0));
    bitmap.insert(1);
    assert_eq!(bitmap.max(), Some(1));
    bitmap.insert(u64::MAX);
    assert_eq!(bitmap.max(), Some(u64::MAX));
}

#[test]
fn test_min() {
    let mut bitmap = RoaringTreemap::new();
    assert_eq!(bitmap.min(), None);
    bitmap.insert(u64::MAX);
    assert_eq!(bitmap.min(), Some(u64::MAX));
    bitmap.insert(1);
    assert_eq!(bitmap.min(), Some(1));
    bitmap.insert(0);
    assert_eq!(bitmap.min(), Some(0));
}

#[test]
fn to_bitmap() {
    let bitmap = (0..5000).collect::<RoaringTreemap>();
    assert_eq!(bitmap.len(), 5000);
    for i in 1..5000 {
        assert!(bitmap.contains(i));
    }
    assert!(!bitmap.contains(5001));
}

#[test]
fn to_array() {
    let mut bitmap = (0..5000).collect::<RoaringTreemap>();
    for i in 3000..5000 {
        bitmap.remove(i);
    }
    assert_eq!(bitmap.len(), 3000);
    for i in 0..3000 {
        assert!(bitmap.contains(i));
    }
    for i in 3000..5000 {
        assert!(!bitmap.contains(i));
    }
}
