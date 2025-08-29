use bip39::Language;

fn main() {
    let words = Language::English.word_list();
    
    println!("Looking for words similar to 'honor' and 'peak'...");
    
    // Find words similar to 'honor'
    let honor_candidates: Vec<&str> = words.iter()
        .filter(|word| word.starts_with("hon") || word.contains("hor"))
        .copied()
        .collect();
    println!("Words similar to 'honor': {:?}", honor_candidates);
    
    // Find words similar to 'peak'  
    let peak_candidates: Vec<&str> = words.iter()
        .filter(|word| word.starts_with("pea") || word.starts_with("pe"))
        .copied()
        .collect();
    println!("Words similar to 'peak': {:?}", peak_candidates);
    
    // Just to verify, let's also check horse and hope which should be valid
    println!("'horse' is valid: {}", words.contains(&"horse"));
    println!("'hope' is valid: {}", words.contains(&"hope"));
    println!("'pear' is valid: {}", words.contains(&"pear"));
    println!("'peace' is valid: {}", words.contains(&"peace"));
}