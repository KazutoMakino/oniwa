//! Open Source Basic Algorithm Code Ingestion Pipeline
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! Retrieves and cleanses basic algorithm and data structure code in Python and Rust
//! under clean open-source licenses (MIT / Apache-2.0 / CC0),
//! recording full provenance to cultivate code syntax and reasoning capabilities.

use oniwa_lm::logger::{DataIngestionLog, ProvenanceEvent, ProvenanceLedger};
use oniwa_lm::reproducibility::compute_checksum_bytes;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct CodeSnippetTarget {
    pub name: &'static str,
    pub title: &'static str,
    pub language: &'static str,
    pub license: &'static str,
    pub description: &'static str,
    pub url: Option<&'static str>,
    pub seed_code: &'static str,
}

pub const DEFAULT_CODE_TARGETS: &[CodeSnippetTarget] = &[
    // --- Python Algorithms ---
    CodeSnippetTarget {
        name: "python_fibonacci",
        title: "Python: Fibonacci Sequence",
        language: "python",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Fibonacci sequence calculation via recursion and dynamic programming",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/fibonacci.py"),
        seed_code: r#"def fibonacci_recursive(n: int) -> int:
    """Return the nth Fibonacci number using recursion."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Return the nth Fibonacci number using iteration (O(n) time, O(1) space)."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


if __name__ == "__main__":
    assert fibonacci_iterative(0) == 0
    assert fibonacci_iterative(1) == 1
    assert fibonacci_iterative(10) == 55
"#,
    },
    CodeSnippetTarget {
        name: "python_factorial",
        title: "Python: Factorial Calculation",
        language: "python",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Factorial calculation via recursion and loops",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/factorial.py"),
        seed_code: r#"def factorial_iterative(n: int) -> int:
    """Compute n! iteratively."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def factorial_recursive(n: int) -> int:
    """Compute n! recursively."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)


if __name__ == "__main__":
    assert factorial_iterative(5) == 120
    assert factorial_recursive(5) == 120
"#,
    },
    CodeSnippetTarget {
        name: "python_binary_search",
        title: "Python: Binary Search Algorithm",
        language: "python",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Binary search on sorted arrays",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/binary_search.py"),
        seed_code: r#"def binary_search(array: list[int], target: int) -> int:
    """Return index of target in sorted array, or -1 if not found."""
    low = 0
    high = len(array) - 1

    while low <= high:
        mid = (low + high) // 2
        guess = array[mid]
        if guess == target:
            return mid
        elif guess < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


if __name__ == "__main__":
    items = [1, 3, 5, 7, 9, 11, 13]
    assert binary_search(items, 7) == 3
    assert binary_search(items, 2) == -1
"#,
    },
    CodeSnippetTarget {
        name: "python_bubble_sort",
        title: "Python: Bubble Sort",
        language: "python",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Bubble sort ascending sorting algorithm",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/bubble_sort.py"),
        seed_code: r#"def bubble_sort(array: list[int]) -> list[int]:
    """Sort a list in ascending order using bubble sort."""
    arr = list(array)
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


if __name__ == "__main__":
    assert bubble_sort([64, 34, 25, 12, 22, 11, 90]) == [11, 12, 22, 25, 34, 64, 90]
"#,
    },
    CodeSnippetTarget {
        name: "python_gcd_lcm",
        title: "Python: Greatest Common Divisor & LCM",
        language: "python",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Greatest common divisor and least common multiple via Euclidean algorithm",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/greatest_common_divisor.py"),
        seed_code: r#"def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor using Euclidean algorithm."""
    while b != 0:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    """Calculate least common multiple."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


if __name__ == "__main__":
    assert gcd(48, 18) == 6
    assert lcm(12, 18) == 36
"#,
    },
    CodeSnippetTarget {
        name: "python_primes",
        title: "Python: Prime Check & Sieve of Eratosthenes",
        language: "python",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Primality test and Sieve of Eratosthenes prime generation",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/prime_check.py"),
        seed_code: r#"def is_prime(n: int) -> bool:
    """Check if an integer is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def sieve_of_eratosthenes(limit: int) -> list[int]:
    """Generate all prime numbers up to limit."""
    if limit < 2:
        return []
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_p[p]:
            for multiple in range(p * p, limit + 1, p):
                is_p[multiple] = False
    return [p for p in range(limit + 1) if is_p[p]]


if __name__ == "__main__":
    assert is_prime(29) is True
    assert is_prime(30) is False
    assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]
"#,
    },
    CodeSnippetTarget {
        name: "python_stack_queue",
        title: "Python: Stack & Queue Data Structures",
        language: "python",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Stack and Queue data structures implementation using list and deque",
        url: None,
        seed_code: r#"class Stack:
    """LIFO Stack data structure."""
    def __init__(self):
        self.items: list[int] = []

    def push(self, item: int) -> None:
        self.items.append(item)

    def pop(self) -> int:
        if not self.items:
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self) -> int:
        if not self.items:
            raise IndexError("peek from empty stack")
        return self.items[-1]

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def __len__(self) -> int:
        return len(self.items)


if __name__ == "__main__":
    s = Stack()
    s.push(10)
    s.push(20)
    assert s.peek() == 20
    assert s.pop() == 20
    assert s.pop() == 10
    assert s.is_empty()
"#,
    },

    // --- Rust Algorithms ---
    CodeSnippetTarget {
        name: "rust_fibonacci",
        title: "Rust: Fibonacci Sequence",
        language: "rust",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Fibonacci implementation in Rust using iterators and recursion",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Rust/master/src/math/fibonacci.rs"),
        seed_code: r#"pub fn fibonacci_recursive(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2),
    }
}

pub fn fibonacci_iterative(n: u32) -> u64 {
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci_iterative(0), 0);
        assert_eq!(fibonacci_iterative(1), 1);
        assert_eq!(fibonacci_iterative(10), 55);
    }
}
"#,
    },
    CodeSnippetTarget {
        name: "rust_binary_search",
        title: "Rust: Binary Search",
        language: "rust",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Generic binary search on Rust slices",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Rust/master/src/searching/binary_search.rs"),
        seed_code: r#"pub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let mut low = 0;
    let mut high = slice.len();

    while low < high {
        let mid = low + (high - low) / 2;
        if &slice[mid] == target {
            return Some(mid);
        } else if &slice[mid] < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_search() {
        let nums = [1, 3, 5, 7, 9, 11];
        assert_eq!(binary_search(&nums, &7), Some(3));
        assert_eq!(binary_search(&nums, &2), None);
    }
}
"#,
    },
    CodeSnippetTarget {
        name: "rust_bubble_sort",
        title: "Rust: Bubble Sort",
        language: "rust",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "In-place bubble sort in Rust",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Rust/master/src/sorting/bubble_sort.rs"),
        seed_code: r#"pub fn bubble_sort<T: Ord>(slice: &mut [T]) {
    let len = slice.len();
    for i in 0..len {
        let mut swapped = false;
        for j in 0..len.saturating_sub(i + 1) {
            if slice[j] > slice[j + 1] {
                slice.swap(j, j + 1);
                swapped = true;
            }
        }
        if !swapped {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_sort() {
        let mut data = [64, 34, 25, 12, 22, 11, 90];
        bubble_sort(&mut data);
        assert_eq!(data, [11, 12, 22, 25, 34, 64, 90]);
    }
}
"#,
    },
    CodeSnippetTarget {
        name: "rust_gcd",
        title: "Rust: Greatest Common Divisor",
        language: "rust",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Greatest common divisor via Euclidean algorithm in Rust",
        url: Some("https://raw.githubusercontent.com/TheAlgorithms/Rust/master/src/math/greatest_common_divisor.rs"),
        seed_code: r#"pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a * b) / gcd(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(lcm(12, 18), 36);
    }
}
"#,
    },
    CodeSnippetTarget {
        name: "rust_stack",
        title: "Rust: Generic Stack",
        language: "rust",
        license: "MIT / CC0 (Clean Algorithm Implementation)",
        description: "Generic Stack struct wrapping Vec in Rust",
        url: None,
        seed_code: r#"#[derive(Debug, Default)]
pub struct Stack<T> {
    elements: Vec<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Self {
        Self { elements: Vec::new() }
    }

    pub fn push(&mut self, item: T) {
        self.elements.push(item);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.elements.pop()
    }

    pub fn peek(&self) -> Option<&T> {
        self.elements.last()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        assert_eq!(s.peek(), Some(&2));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
        assert!(s.is_empty());
    }
}
"#,
    },
];

pub struct CodePipeline {
    #[allow(dead_code)]
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub corpus_dir: PathBuf,
    pub raw_dir: PathBuf,
    pub force_download: bool,
}

impl CodePipeline {
    pub fn new<P: AsRef<Path>>(data_dir: P, logs_dir: P) -> Self {
        let data_p = data_dir.as_ref().to_path_buf();
        let logs_p = logs_dir.as_ref().to_path_buf();
        let corpus_dir = data_p.join("corpus");
        let raw_dir = data_p.join("raw");

        fs::create_dir_all(&corpus_dir).ok();
        fs::create_dir_all(&raw_dir).ok();

        Self {
            data_dir: data_p,
            logs_dir: logs_p,
            corpus_dir,
            raw_dir,
            force_download: false,
        }
    }

    pub fn set_force(&mut self, force: bool) {
        self.force_download = force;
    }

    /// Retrieve a single code target or deploy from embedded seeds
    pub fn fetch_or_seed(
        &self,
        target: &CodeSnippetTarget,
    ) -> Result<(String, String), Box<dyn std::error::Error>> {
        let ext = if target.language == "python" {
            "py"
        } else {
            "rs"
        };
        let raw_filename = format!("code_{}.{}", target.name, ext);
        let raw_path = self.raw_dir.join(&raw_filename);

        // Use cache if available unless force_download is set
        if !self.force_download && raw_path.exists() && raw_path.metadata()?.len() > 0 {
            let content = fs::read_to_string(&raw_path)?;
            let sha = compute_checksum_bytes(content.as_bytes());
            return Ok((content, sha));
        }

        // Attempt download if URL is provided
        if let Some(url) = target.url {
            println!("  📥 [Code] Attempting download: \"{}\" ...", target.title);
            let user_agent =
                "oniwa-pipeline/0.1.0 (Ethical Open Source Code Ingestion; Educational AI)";
            let status = Command::new("curl")
                .arg("-s")
                .arg("-f")
                .arg("-L")
                .arg("-m")
                .arg("5") // 5-second timeout
                .arg("-A")
                .arg(user_agent)
                .arg("-o")
                .arg(&raw_path)
                .arg(url)
                .status();

            if let Ok(st) = status {
                if st.success() && raw_path.exists() && raw_path.metadata()?.len() > 0 {
                    let content = fs::read_to_string(&raw_path)?;
                    let sha = compute_checksum_bytes(content.as_bytes());
                    sleep(Duration::from_millis(500));
                    return Ok((content, sha));
                }
            }
            println!(
                "  ⚠️ Download failed or timed out. Deploying high-quality seed code: \"{}\"",
                target.title
            );
        }

        // Fallback: write seed code
        fs::write(&raw_path, target.seed_code.as_bytes())?;
        let sha = compute_checksum_bytes(target.seed_code.as_bytes());
        Ok((target.seed_code.to_string(), sha))
    }

    /// Preprocess single code file, save to corpus, and log to provenance ledger
    pub fn ingest_single_code(
        &self,
        target: &CodeSnippetTarget,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let (raw_content, raw_sha) = self.fetch_or_seed(target)?;

        // Code cleansing and normalization
        let cleaned_code = clean_code(&raw_content, target.language);

        let out_filename = format!("code_{}.txt", target.name);
        let out_path = self.corpus_dir.join(&out_filename);
        fs::write(&out_path, cleaned_code.as_bytes())?;

        let clean_sha = compute_checksum_bytes(cleaned_code.as_bytes());
        let char_count = cleaned_code.chars().count();

        println!(
            "  💻 Saved code: \"{}\" ({}, {} characters / {:.2} KB)",
            target.title,
            target.language,
            char_count,
            cleaned_code.len() as f32 / 1024.0
        );

        let ledger_path = self.logs_dir.join("ledger_index.jsonl");
        let mut ledger = ProvenanceLedger::open(&ledger_path)?;

        ledger.record(&ProvenanceEvent::DataIngestion(DataIngestionLog {
            timestamp_utc: oniwa_lm::logger::current_timestamp_utc(),
            source_name: format!("CleanCode: \"{}\" ({})", target.title, target.language),
            source_url_or_path: target.url.unwrap_or("builtin-seed").to_string(),
            license: target.license.to_string(),
            raw_data_sha256: raw_sha,
            raw_data_bytes: raw_content.len(),
            tokenized_sha256: clean_sha,
            num_tokens: char_count,
            vocab_size: 0,
            tokenizer_type: "Raw Cleaned Code (Python/Rust)".to_string(),
        }))?;

        Ok(out_path)
    }

    /// Batch ingest default code targets
    pub fn ingest_default_code(&self) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        println!("============================================================");
        println!(" 💻 Open-Source Basic Algorithm Code Ingestion (Python/Rust)");
        println!("    (Source: MIT / Apache-2.0 / CC0 Open Licenses)");
        println!("============================================================");

        let mut paths = Vec::new();
        for target in DEFAULT_CODE_TARGETS {
            println!(
                "▶ \"{}\" [{}]: {}",
                target.title, target.language, target.description
            );
            let path = self.ingest_single_code(target)?;
            paths.push(path);
        }

        Ok(paths)
    }
}

/// Code normalization and cleansing process
/// - Trim trailing whitespace
/// - Replace tabs with 4 spaces
/// - Compress 3+ consecutive empty lines to 2 empty lines
pub fn clean_code(raw_code: &str, _language: &str) -> String {
    let mut cleaned_lines: Vec<String> = Vec::new();
    let mut consecutive_empty_lines = 0;

    for line in raw_code.lines() {
        // Normalize tabs to 4 spaces and trim trailing whitespace
        let expanded = line.replace('\t', "    ");
        let trimmed_end = expanded.trim_end();

        if trimmed_end.is_empty() {
            consecutive_empty_lines += 1;
            if consecutive_empty_lines <= 2 {
                cleaned_lines.push(String::new());
            }
        } else {
            consecutive_empty_lines = 0;
            cleaned_lines.push(trimmed_end.to_string());
        }
    }

    let mut result = cleaned_lines.join("\n");
    result.push('\n');
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_code_normalization() {
        let raw = "def foo():\n\tprint(\"hello\")   \n\n\n\n\treturn 42   \n";
        let cleaned = clean_code(raw, "python");
        assert!(cleaned.contains("    print(\"hello\")"));
        assert!(cleaned.contains("    return 42"));
        // Verify no trailing whitespace
        for line in cleaned.lines() {
            assert_eq!(line, line.trim_end());
        }
        // Verify consecutive empty lines are capped at 2
        assert!(!cleaned.contains("\n\n\n\n"));
    }

    #[test]
    fn test_default_code_targets_validity() {
        assert!(!DEFAULT_CODE_TARGETS.is_empty());
        for target in DEFAULT_CODE_TARGETS {
            assert!(!target.name.is_empty());
            assert!(!target.seed_code.is_empty());
            assert!(target.language == "python" || target.language == "rust");
            let cleaned = clean_code(target.seed_code, target.language);
            assert!(!cleaned.is_empty());
        }
    }
}
