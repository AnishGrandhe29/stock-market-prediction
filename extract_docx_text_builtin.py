
import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_builtin(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
        
        tree = ET.fromstring(xml_content)
        
        # Namespace map in docx xml
        namespaces = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        }
        
        text_parts = []
        
        # Iterate over all paragraph elements (w:p)
        for p in tree.findall('.//w:p', namespaces):
            # Check if this paragraph is inside a table cell (w:tc) -> actually we just flatten everything first then format
            # A better way is to iterate over body elements
            
            # Let's just extract all text nodes (w:t)
            # This ignores structure but gives us the content to analyze
            texts = [node.text for node in p.findall('.//w:t', namespaces) if node.text]
            if texts:
                text_parts.append(''.join(texts))
        
        return '\n'.join(text_parts)
    except Exception as e:
        return f"Error reading docx: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx_text_builtin.py <docx_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(extract_text_builtin(path))
