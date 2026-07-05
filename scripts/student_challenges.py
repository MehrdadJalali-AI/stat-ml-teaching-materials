"""Add Week-2-style in-class student code challenges to notebook builders."""
import nbformat as nbf

CHALLENGE_INTRO = (
    "Do these in class after reviewing the notebook code. "
    "Change the code, rerun the cell, and compare the output.\n\n"
    "The cells below are exercises and are left for you to complete."
)


def student_challenge_cells(challenges):
    """Return markdown/code cells for 3 (or more) student challenges.

    Each challenge is (title, question, [todo, todo, ...]).
    """
    cells = [nbf.v4.new_markdown_cell(f"## Student Code Challenges\n\n{CHALLENGE_INTRO}")]
    for index, (title, question, todos) in enumerate(challenges, start=1):
        cells.append(
            nbf.v4.new_markdown_cell(
                f"### Challenge {index} - {title}\n\n**Question:** {question}\n\n**Code:**"
            )
        )
        todo_code = "\n".join(f"# TODO: {line}" for line in todos) + "\n"
        cells.append(nbf.v4.new_code_cell(todo_code))
    return cells
