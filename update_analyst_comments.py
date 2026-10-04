import os
import glob

# Audit all Python files in target directory and count lines & comments
def audit_python_files(target_dir="."):
    python_files = glob.glob(os.path.join(target_dir, "*.py")) + glob.glob(os.path.join(target_dir, "*.PY"))
    python_files = sorted(list(set(os.path.abspath(f) for f in python_files)))
    
    print(f"Auditing {len(python_files)} Python files in {os.path.abspath(target_dir)}:\n")
    for file_path in python_files:
        name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        comments = sum(1 for line in lines if line.strip().startswith('#'))
        print(f" - {name:<25} | Total lines: {len(lines):<4} | Comments: {comments}")

if __name__ == "__main__":
    audit_python_files()
