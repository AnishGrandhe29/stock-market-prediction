import zipfile
import xml.etree.ElementTree as ET
import sys
import os


def extract_comments_with_context(docx_path):
    comments_map = {}
    
    try:
        with zipfile.ZipFile(docx_path, 'r') as z:
            # 1. Read comments.xml to get id -> comment text mapping
            try:
                comments_xml = z.read('word/comments.xml')
            except KeyError:
                print("No comments found in the document.")
                return []
            
            et = ET.fromstring(comments_xml)
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            for comment in et.findall('.//w:comment', namespace):
                comment_id = comment.get(f"{{{namespace['w']}}}id")
                author = comment.get(f"{{{namespace['w']}}}author")
                
                text_parts = []
                for p in comment.findall('.//w:p', namespace):
                    for r in p.findall('.//w:r', namespace):
                        t = r.find('w:t', namespace)
                        if t is not None and t.text:
                            text_parts.append(t.text)
                
                comment_text = "".join(text_parts)
                comments_map[comment_id] = {'author': author, 'text': comment_text, 'context': []}

            # 2. Read document.xml to find where comments are referenced
            document_xml = z.read('word/document.xml')
            doc_root = ET.fromstring(document_xml)
            
            # Find paragraphs
            for p in doc_root.findall('.//w:p', namespace):
                p_text = ""
                # Get text of the paragraph
                for r in p.findall('.//w:r', namespace):
                    t = r.find('w:t', namespace)
                    if t is not None and t.text:
                        p_text += t.text
                
                # Check for comment references in this paragraph
                # rangeStart and rangeEnd are also used but commentReference is the anchor often used for simple comments
                # Iterate all elements to find comment references in order
                for child in p.iter():
                    if child.tag == f"{{{namespace['w']}}}commentReference":
                        cid = child.get(f"{{{namespace['w']}}}id")
                        if cid in comments_map:
                            comments_map[cid]['context'].append(p_text[:200]) # Store start of paragraph as context

                    elif child.tag == f"{{{namespace['w']}}}commentRangeStart":
                         cid = child.get(f"{{{namespace['w']}}}id")
                         if cid in comments_map:
                            comments_map[cid]['context'].append(p_text[:200])

    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    results = []
    for cid, data in comments_map.items():
        if data['text'].strip():
             context_str = " | ".join(set(data['context']))
             results.append(f"Author: {data['author']}\nContext (Paragraph): {context_str}\nComment: {data['text']}\n---")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_comments.py <path_to_docx>")
        sys.exit(1)
        
    docx_file = sys.argv[1]
    if not os.path.exists(docx_file):
        print(f"File not found: {docx_file}")
        sys.exit(1)
        
    comments = extract_comments_with_context(docx_file)
    for c in comments:
        print(c)

