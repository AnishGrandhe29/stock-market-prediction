
import os

def generate_co_po_doc(output_path):
    # content definitions
    project_title = "Multimodal NIFTY-50 Market Prediction System using Deep Learning"
    
    # Standard Engineering POs (simplified for the doc)
    pos = [
        "PO1: Engineering Knowledge: Apply the knowledge of mathematics, science, engineering fundamentals, and an engineering specialization to the solution of complex engineering problems.",
        "PO2: Problem Analysis: Identify, formulate, review research literature, and analyze complex engineering problems reaching substantiated conclusions using first principles of mathematics, natural sciences, and engineering sciences.",
        "PO3: Design/Development of Solutions: Design solutions for complex engineering problems and design system components or processes that meet the specified needs with appropriate consideration for the public health and safety, and the cultural, societal, and environmental considerations.",
        "PO4: Conduct Investigations of Complex Problems: Use research-based knowledge and research methods including design of experiments, analysis and interpretation of data, and synthesis of the information to provide valid conclusions.",
        "PO5: Modern Tool Usage: Create, select, and apply appropriate techniques, resources, and modern engineering and IT tools including prediction and modeling to complex engineering activities with an understanding of the limitations.",
        "PO6: The Engineer and Society: Apply reasoning informed by the contextual knowledge to assess societal, health, safety, legal and cultural issues and the consequent responsibilities relevant to the professional engineering practice.",
        "PO7: Environment and Sustainability: Understand the impact of the professional engineering solutions in societal and environmental contexts, and demonstrate the knowledge of, and need for sustainable development.",
        "PO8: Ethics: Apply ethical principles and commit to professional ethics and responsibilities and norms of the engineering practice.",
        "PO9: Individual and Team Work: Function effectively as an individual, and as a member or leader in diverse teams, and in multidisciplinary settings.",
        "PO10: Communication: Communicate effectively on complex engineering activities with the engineering community and with society at large, such as, being able to comprehend and write effective reports and design documentation, make effective presentations, and give and receive clear instructions.",
        "PO11: Project Management and Finance: Demonstrate knowledge and understanding of the engineering and management principles and apply these to one’s own work, as a member and leader in a team, to manage projects and in multidisciplinary environments.",
        "PO12: Life-long Learning: Recognize the need for, and have the preparation and ability to engage in independent and life-long learning in the broadest context of technological change."
    ]

    # Course Outcomes tailored to the project
    cos = [
        "CO1: Understand and analyze the complexities of financial time-series data and the limitations of existing unimodal prediction models.",
        "CO2: Apply data preprocessing techniques, including normalization and technical indicator calculation, to prepare multimodal datasets (numerical and textual).",
        "CO3: Design and implement a deep learning-based multimodal architecture using Temporal Convolutional Networks (TCN) and Sentiment Analysis for stock market prediction.",
        "CO4: Evaluate model performance using standard metrics (RMSE, MAPE) and interpret predictions using Explainable AI (SHAP) techniques."
    ]

    # Mapping Matrix (3=High, 2=Medium, 1=Low, -=No correlation)
    # CO1 (Analysis): Strongly maps to PO1 (Knowledge), PO2 (Analysis), PO4 (Investigation)
    # CO2 (Preprocessing): PO2 (Analysis), PO3 (Design), PO5 (Tools)
    # CO3 (Implementation): PO3 (Design), PO5 (Tools), PO9 (Teamwork - implied project work)
    # CO4 (Evaluation): PO4 (Investigation), PO5 (Tools), PO10 (Communication needed for explanation)
    
    mapping = [
        # PO1 PO2 PO3 PO4 PO5 PO6 PO7 PO8 PO9 PO10 PO11 PO12
        ["CO1", "3", "3", "-", "2", "2", "-", "-", "-", "-", "-", "-", "2"],
        ["CO2", "3", "2", "3", "-", "3", "-", "-", "-", "-", "-", "-", "-"],
        ["CO3", "3", "3", "3", "2", "3", "-", "-", "-", "2", "-", "-", "2"],
        ["CO4", "2", "3", "-", "3", "3", "1", "-", "1", "-", "2", "-", "2"]
    ]

    # HTML Template
    html_content = f"""
    <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
    <head>
    <meta charset="utf-8">
    <title>{project_title}</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; font-size: 12pt; margin: 1in; }}
        h1, h2, h3 {{ text-align: center; color: #2E74B5; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid black; padding: 8px; text-align: left; vertical-align: top; }}
        th {{ background-color: #F2F2F2; font-weight: bold; text-align: center; }}
        .center {{ text-align: center; }}
    </style>
    </head>
    <body>

    <h1>Course Outcomes and Program Outcomes Mapping</h1>
    <h2 style="color:black;">Project Title: {project_title}</h2>
    
    <h3>1. Course Outcomes (COs)</h3>
    <p>After completing this project, the students will be able to:</p>
    <table>
        <tr>
            <th width="15%">CO ID</th>
            <th>Description</th>
        </tr>
        {"".join([f"<tr><td class='center'>{co.split(':')[0]}</td><td>{co.split(':')[1]}</td></tr>" for co in cos])}
    </table>

    <h3>2. Program Outcomes (POs)</h3>
    <table>
        <tr>
            <th width="15%">PO ID</th>
            <th>Description</th>
        </tr>
        {"".join([f"<tr><td class='center'>{po.split(':')[0]}</td><td>{po.split(':')[1]}</td></tr>" for po in pos])}
    </table>

    <h3>3. CO-PO Mapping Matrix</h3>
    <p>Mapping Level: 3 = High, 2 = Medium, 1 = Low</p>
    <table>
        <tr>
            <th>CO / PO</th>
            <th>PO1</th><th>PO2</th><th>PO3</th><th>PO4</th><th>PO5</th><th>PO6</th><th>PO7</th><th>PO8</th><th>PO9</th><th>PO10</th><th>PO11</th><th>PO12</th>
        </tr>
        {"".join([f"<tr>{''.join([f'<td class=center>{cell}</td>' for cell in row])}</tr>" for row in mapping])}
    </table>

    <h3>4. Justification for Mapping</h3>
    <p>
    <strong>CO1-PO1/PO2:</strong> Requires deep understanding of mathematical foundations of deep learning and financial analysis.<br>
    <strong>CO2-PO3/PO5:</strong> Involves designing data pipelines using modern tools like Pandas and TA-Lib.<br>
    <strong>CO3-PO3/PO5:</strong> Application of complex engineering solutions (TCNs) using PyTorch.<br>
    <strong>CO4-PO4:</strong> Investigating model behavior and validating results against real-world data.
    </p>

    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated {output_path}")

if __name__ == "__main__":
    output_file = r"a:\project\project-final-anish\stock-market-prediction\Documentation\Project_CO_PO.doc"
    generate_co_po_doc(output_file)
