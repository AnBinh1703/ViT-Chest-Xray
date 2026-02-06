"""
Script to format all section/subsection/subsubsection headings in LaTeX document
Makes all headings bold according to academic standards
"""
import re

def format_latex_headings(input_file, output_file=None):
    """
    Format all LaTeX headings to be bold
    Handles: \section{}, \subsection{}, \subsubsection{}
    Skips: \section*, \subsection*, etc. (starred versions)
    """
    if output_file is None:
        output_file = input_file
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match section commands that DON'T already have \textbf
    # This avoids double-wrapping
    patterns = {
        'section': r'\\section\{([^}]+)\}',
        'subsection': r'\\subsection\{([^}]+)\}',
        'subsubsection': r'\\subsubsection\{([^}]+)\}'
    }
    
    changes_made = 0
    
    for cmd_type, pattern in patterns.items():
        # Find all matches
        matches = list(re.finditer(pattern, content))
        
        # Process in reverse order to preserve positions
        for match in reversed(matches):
            title = match.group(1)
            
            # Skip if already has \textbf at the start
            if title.strip().startswith('\\textbf{'):
                continue
            
            # Create replacement with bold
            old_text = match.group(0)
            new_text = f'\\{cmd_type}{{\\textbf{{{title}}}}}'
            
            # Replace in content
            start, end = match.span()
            content = content[:start] + new_text + content[end:]
            changes_made += 1
            print(f"✓ Formatted: \\{cmd_type}{{{title[:50]}...}}")
    
    # Write back
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n{'='*60}")
    print(f"Total changes made: {changes_made}")
    print(f"Output written to: {output_file}")
    print(f"{'='*60}")
    
    return changes_made

if __name__ == "__main__":
    input_file = r"d:\MSE\10.Deep Learning\Group_Final\ViT-Chest-Xray\Report\LaTeX\final.tex"
    
    print("Formatting LaTeX headings...")
    print(f"Input file: {input_file}\n")
    
    changes = format_latex_headings(input_file)
    
    if changes > 0:
        print(f"\n✓ Successfully formatted {changes} headings!")
    else:
        print("\n⚠ No changes needed - all headings already formatted.")
