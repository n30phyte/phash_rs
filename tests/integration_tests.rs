use std::path::PathBuf;

use phash_rs::Image;

#[test]
fn test_phash() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests/test_cases");
    d.push("Alyson_Hannigan_200512.jpg");
    let image = Image::new(d);
    let hash = image.unwrap().calculate_hash();

    assert_eq!(format!("{:x}", hash), "f4b883c6338bc3cc");
}
