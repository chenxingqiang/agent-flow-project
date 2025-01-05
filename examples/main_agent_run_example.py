from examples.academic_agent import AcademicAgent

def create_initial_input():
    return {
        'student_needs': {
            'research_topic': 'AI Applications in Education',
            'deadline': '2024-12-31',
            'academic_level': 'PhD',
            'field': 'Computer Science',
            'special_requirements': 'Focus on machine learning applications',
            'author': 'John Doe'
        }
    }

def generate_documents(agent, results):
    pdf_path = agent.generate_output_document(
        results, 
        'pdf', 
        'output/research_plan.pdf'
    )
    
    docx_path = agent.generate_output_document(
        results, 
        'docx', 
        'output/research_plan.docx'
    )
    
    markdown_path = agent.generate_output_document(
        results, 
        'markdown', 
        'output/research_plan.md'
    )
    
    print("Generated documents:")
    print(f"PDF: {pdf_path}")
    print(f"Word: {docx_path}")
    print(f"Markdown: {markdown_path}")

def main():
    agent = AcademicAgent('config.json', 'data/agent.json')
    try:
        results = agent.execute_workflow(create_initial_input())
        generate_documents(agent, results)
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()