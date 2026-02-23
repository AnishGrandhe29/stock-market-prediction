
import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_structure(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
        
        tree = ET.fromstring(xml_content)
        
        namespaces = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        }
        
        output = []
        
        # We need to iterate over the body's children to keep order of tables and paragraphs
        body = tree.find('w:body', namespaces)
        if body is None:
            return "No body found in docx"
            
        for child in body:
            tag = child.tag
            if tag.endswith('p'): # Paragraph
                texts = [node.text for node in child.findall('.//w:t', namespaces) if node.text]
                if texts:
                    output.append("PARA: " + ''.join(texts))
            elif tag.endswith('tbl'): # Table
                output.append("TABLE START")
                for row in child.findall('.//w:tr', namespaces):
                    row_data = []
                    for cell in row.findall('.//w:tc', namespaces):
                        # Cells can contain paragraphs
                        cell_texts = [node.text for node in cell.findall('.//w:t', namespaces) if node.text]
                        row_data.append(''.join(cell_texts))
                    output.append("  ROW: | " + " | ".join(row_data) + " |")
                output.append("TABLE END")
        
        return '\n'.join(output)
    except Exception as e:
        return f"Error reading docx: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx_structure.py <docx_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(extract_structure(path))
