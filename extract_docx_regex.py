
import zipfile
import re
import sys

def extract_regex(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml').decode('utf-8')
        
        # Regex for w:t
        # <w:t>Text</w:t> or <w:t xml:space="preserve">Text</w:t>
        matches = re.findall(r'<w:t[^>]*>(.*?)</w:t>', xml_content)
        return '\n'.join(matches)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    path = sys.argv[1]
    print(extract_regex(path))
