# Movie_Recommendation_System

Here’s a **clean, professional `README.md` text** you can directly copy and use for your project 👌
(It’s written for **GitHub + Jupyter Notebook users** and matches your code perfectly.)

---

# 🎬 Movie Recommendation System (Jupyter Notebook)

A **content-based movie recommendation system** built using **Python, Pandas, and Scikit-learn**.
It recommends movies based on **plot similarity** using **TF-IDF vectorization** and **cosine similarity**, with a **beautiful styled output** inside Jupyter Notebook.

---

## 📌 Features

* 🔍 Search movies by title
* 🎯 Get **Top 5 similar movies** instantly
* 🧠 Content-based filtering using movie overviews
* 🎨 Attractive UI using **HTML + CSS** inside Jupyter
* ⚠️ Handles:

  * Empty input
  * Wrong movie names
  * Suggests close matches

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **Jupyter Notebook**
* **HTML & CSS (IPython Display)**

---

## 📂 Dataset

* **TMDB 5000 Movies Dataset**
* File used:

  ```
  tmdb_5000_movies.csv
  ```

**Required columns:**

* `title`
* `overview`



## ▶️ How to Run 

1. Open **Jupyter Notebook**
2. Open the notebook file
3. Run all cells
4. Use the function below 👇

```python
recommend_movies('Inception')
```

---

## 💡 Example Inputs

Try searching with:

* The Dark Knight
* Avatar
* Inception
* Interstellar
* The Matrix

---

## 📊 How It Works (Concept)

1. Movie overviews are converted into vectors using **TF-IDF**
2. **Cosine similarity** measures similarity between movies
3. Top 5 most similar movies are returned
4. Results are displayed using styled **HTML cards**






## 🎬 Movie Recommendation System – Streamlit Web App (Additional Work Done)

This is an **interactive web-based version** of the Movie Recommendation System built using **Streamlit**.
It extends the notebook-based assignment into a **modern UI web application** for real-time movie recommendations.

---

## 🚀 Features

* 🔍 Search movies by title
* 🎯 Get **Top 5 similar movies** instantly
* 🎨 Clean, responsive UI with custom CSS
* ⚠️ Error handling with smart suggestions
* 📊 Sidebar with system stats & usage tips

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas**
* **Scikit-learn**
* **TF-IDF + Cosine Similarity**

---

## 📂 Dataset

* **tmdb_5000_movies.csv**
* Columns used: `title`, `overview`

Dataset must be placed in the **same directory** as `app.py`.

---

## ▶️ How to Run

```bash
pip install streamlit pandas scikit-learn
streamlit run app.py
```



