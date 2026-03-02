import os
import site

print("--- Searching for pandas_ta file ---")

# Get all possible site-packages folders in the venv
site_packages = site.getsitepackages()

file_found = False

for path in site_packages:
    print(f"Scanning: {path} ...")
    for root, dirs, files in os.walk(path):
        if "squeeze_pro.py" in files:
            full_path = os.path.join(root, "squeeze_pro.py")
            print(f"\nFOUND FILE: {full_path}")
            
            try:
                # Read the file
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Check for the bug
                if "from numpy import NaN as npNaN" in content:
                    print("Bug detected. Applying fix...")
                    # Replace the bad code
                    new_content = content.replace("from numpy import NaN as npNaN", "from numpy import nan as npNaN")
                    
                    # Write it back
                    with open(full_path, 'w') as f:
                        f.write(new_content)
                    print("SUCCESS: File has been patched!")
                    file_found = True
                
                elif "from numpy import nan as npNaN" in content:
                    print("File is already fixed.")
                    file_found = True
                
                else:
                    print("Could not find the specific line to fix. Please check manually.")
                
            except Exception as e:
                print(f"Error editing file: {e}")
            
            # Stop searching once found
            break
    if file_found:
        break

if not file_found:
    print("\nERROR: Could not find 'squeeze_pro.py'.")
    print("Please ensure pandas_ta is installed via 'python setup.py install'.")