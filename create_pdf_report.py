"""
Script to convert the stakeholder report to PDF format
"""
import os
import subprocess
import sys

def create_pdf():
    """Convert markdown to PDF using available tools"""
    
    markdown_file = "Project_Overview_Stakeholder_Report.md"
    pdf_file = "Australian_Export_Data_Project_Stakeholder_Report.pdf"
    
    print("Creating PDF report for stakeholder...")
    
    # Try different methods to convert to PDF
    try:
        # Method 1: Try using pandoc if available
        result = subprocess.run([
            'pandoc', 
            markdown_file, 
            '-o', pdf_file,
            '--pdf-engine=wkhtmltopdf',
            '--css=style.css'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ PDF created successfully: {pdf_file}")
            return True
        else:
            print("Pandoc not available, trying alternative method...")
            
    except FileNotFoundError:
        print("Pandoc not found, trying alternative method...")
    
    try:
        # Method 2: Try using markdown2pdf
        import markdown2
        from weasyprint import HTML, CSS
        
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        html_content = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks'])
        
        # Add basic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Australian Export Data Project Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                .status {{ background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }}
                ul {{ margin: 10px 0; }}
                li {{ margin: 5px 0; }}
                .emoji {{ font-size: 1.2em; }}
                strong {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        HTML(string=styled_html).write_pdf(pdf_file)
        print(f"✅ PDF created successfully: {pdf_file}")
        return True
        
    except ImportError:
        print("Required libraries not available for PDF conversion.")
        print("Installing required packages...")
        
        # Install required packages
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'markdown2', 'weasyprint'], 
                      capture_output=True)
        
        print("Please run the script again after installation completes.")
        return False
    
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return False

def create_html_version():
    """Create an HTML version that can be easily converted to PDF"""
    
    markdown_file = "Project_Overview_Stakeholder_Report.md"
    html_file = "Project_Overview_Stakeholder_Report.html"
    
    try:
        import markdown2
        
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        html_content = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks'])
        
        # Create styled HTML
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Australian Export Data Project - Stakeholder Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    line-height: 1.6; 
                    margin: 0;
                    padding: 40px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 40px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{ 
                    color: #2c3e50; 
                    border-bottom: 3px solid #3498db; 
                    padding-bottom: 10px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h2 {{ 
                    color: #34495e; 
                    margin-top: 30px; 
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                h3 {{ color: #7f8c8d; margin-top: 20px; }}
                .status {{ 
                    background-color: #d4edda; 
                    padding: 15px; 
                    border-left: 4px solid #28a745; 
                    margin: 15px 0; 
                    border-radius: 4px;
                }}
                ul {{ margin: 15px 0; }}
                li {{ margin: 8px 0; }}
                .emoji {{ font-size: 1.2em; }}
                strong {{ color: #2c3e50; }}
                .highlight {{
                    background-color: #fff3cd;
                    padding: 10px;
                    border-left: 4px solid #ffc107;
                    margin: 10px 0;
                    border-radius: 4px;
                }}
                @media print {{
                    body {{ background-color: white; }}
                    .container {{ box-shadow: none; }}
                }}
            </style>
        </head>
        <body>
        <div class="container">
        {html_content}
        </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(styled_html)
        
        print(f"✅ HTML report created: {html_file}")
        print("You can open this in a web browser and print to PDF")
        return True
        
    except ImportError:
        print("Installing markdown2...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'markdown2'], 
                      capture_output=True)
        print("Please run the script again after installation completes.")
        return False
    except Exception as e:
        print(f"Error creating HTML: {e}")
        return False

if __name__ == "__main__":
    print("Creating stakeholder report...")
    
    # Try to create PDF first
    if not create_pdf():
        print("\nCreating HTML version instead...")
        create_html_version()
    
    print("\n📄 Report creation complete!")
    print("The report is ready for your stakeholder presentation.")


