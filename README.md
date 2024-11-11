
# Movie Recommendation Web App

This is a Flask-based web application that provides movie recommendations using a machine learning model trained on movie data. The model predicts similar movies based on a given movie title.

## Features
- Movie recommendation using a trained `MovieRecommender` model.
- Interactive web interface with a Netflix-style cinematic look.
- Built with Flask, Pandas, and scikit-learn.
- Model is pre-trained and stored in `model.pkl`.

## Project Structure

```
project-root/
├── app.py                     # Flask application file
├── model.pkl                  # Trained model file
├── templates/                 # Folder for HTML templates
│   └── index.html             # Main HTML file for the webpage
├── static/                    # Static assets folder for CSS, JavaScript, and images
│   ├── css/
│   │   └── style.css          # Styling for the web page
│   ├── js/
│   │   └── script.js          # JavaScript for interactions
│   └── img/
│       └── your-hero-image.jpg # Hero image for the landing page
└── README.md                  # Documentation for the project
└── ipynb_to_pkl.ipynb         # Jupyter notebook to convert model.ipynb to model.pkl
```

## Setup and Installation

### Step 1: Install Dependencies

Make sure you have Python 3.x installed. You can create a virtual environment and install the necessary dependencies using the following commands:

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

   **`requirements.txt`** should include:
   ```text
   Flask
   pandas
   scikit-learn
   dill
   ```

### Step 2: Convert Model

To convert the model from `model.ipynb` (your Jupyter notebook) to `model.pkl`, use the provided Jupyter notebook `ipynb_to_pkl.ipynb`. Run this notebook in a Jupyter environment:

1. Open the notebook (`ipynb_to_pkl.ipynb`) in Jupyter.
2. Execute the notebook to load the model from `model.ipynb` and save it as `model.pkl`.

Once you have `model.pkl`, place it in the project root directory.

### Step 3: Run the Flask Application

After setting up your environment and converting the model, run the Flask application:

```bash
python app.py
```

By default, the application will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

### Step 4: Test the Application

To test if the project is running successfully, you can use the following steps:

1. **Open your web browser** and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. **Enter a movie title** in the input field. For example, type:
   ```
   Jumanji (1995)
   ```
3. **Submit** the form to get movie recommendations.

### Sample Input for Testing:

- **Movie Title Input**: `Jumanji (1995)`
- **Expected Output**: A list of recommended movies similar to *Jumanji (1995)*, such as:
  - Zathura: A Space Adventure
  - The Jungle Book
  - Jumanji: Welcome to the Jungle
  - etc.

### Step 5: Access the Model API

If you want to interact with the model API directly, you can use the `/recommend` endpoint.

#### Example API Request

**POST Request** (to `http://127.0.0.1:5000/recommend`):
```json
{
  "movie_title": "Jumanji (1995)"
}
```

#### Expected Response:

```json
{
  "recommendations": [
    "Zathura: A Space Adventure",
    "The Jungle Book",
    "Jumanji: Welcome to the Jungle",
    ...
  ]
}
```

---

## Additional Notes

- The **`model.pkl`** file is essential for the recommendation system to work. Ensure it's generated correctly and placed in the root directory.
- If you modify or update the model, make sure to regenerate `model.pkl` using `ipynb_to_pkl.ipynb`.
- You can also extend the web application with more features like user input validation or additional recommendations.

