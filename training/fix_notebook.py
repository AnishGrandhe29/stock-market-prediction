import json

file_path = r'c:\Users\grand\Desktop\4thyrproject\training\NIFTY50_Complete_Training.ipynb'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Iterate through cells to find the pip install cell
    found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Check if source is a list and has content
            if isinstance(source, list) and len(source) > 0 and source[0].strip().startswith('!pip install'):
                print(f"Found pip install cell: {source[0]}")
                # Replace the line
                source[0] = '!pip install "numpy<2.0" "pandas==2.2.2" yfinance feedparser pandas-ta scikit-learn torch torchvision tqdm matplotlib -q\n'
                print(f"Replaced with: {source[0]}")
                found = True
                break
    
    if found:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print("Notebook updated successfully.")
    else:
        print("Could not find the pip install cell.")

except Exception as e:
    print(f"Error: {e}")
