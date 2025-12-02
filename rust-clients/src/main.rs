mod config; 

fn main() {
    let cfg = config::AppConfig::default(); 

    println!("{:?}", cfg);
    println!("Hello, world!");
}
