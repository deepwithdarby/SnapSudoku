# Sudoku Solver - Hugging Face Space

This is a Sudoku solver application that can be deployed as a Hugging Face Space.

## How to deploy

To deploy this application on Hugging Face Spaces, you need to upload the following files to your space:

1.  `app.py`: The main application file with the Gradio interface.
2.  `requirements.txt`: The file with the required Python dependencies.
3.  The `networks` directory, containing the pre-trained neural network model `net`.

**Important:** The `networks/net` file is crucial for the application to work. It contains the pre-trained model for digit recognition. Without this file, the application will not be able to solve the Sudoku puzzles.

Once you have uploaded these files, your Hugging Face Space should be up and running.
