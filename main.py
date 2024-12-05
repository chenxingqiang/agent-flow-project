from examples.academic_agent import AcademicAgent

def main():
    # Initialize agent
    agent = AcademicAgent('config.json', 'data/agent.json')
    
    # Execute workflow
    initial_input = {
        'student_needs': {
            'research_topic': 'AI Applications in Education',
            'deadline': '2024-12-31',
            'academic_level': 'PhD',
            'field': 'Computer Science',
            'special_requirements': 'Focus on machine learning applications',
            'author': 'John Doe'
        }
    }
    
    try:
        # Execute workflow
        results = agent.execute_workflow(initial_input)
        
        # Generate documents in different formats
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
        
        print(f"Generated documents:")
        print(f"PDF: {pdf_path}")
        print(f"Word: {docx_path}")
        print(f"Markdown: {markdown_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
if __name__ == "__main__":
    main()