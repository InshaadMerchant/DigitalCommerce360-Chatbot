# DigitalCommerce360 Chatbot

An intelligent chatbot built with Flask that answers questions using **DigitalCommerce360** data.
Designed to provide instant insights on e-commerce trends, companies, and reports from the DigitalCommerce360 dataset.

## Features

* 💬 Conversational interface for DigitalCommerce360 data
* 🔍 Search and respond to domain-specific queries
* ⚡ Fast and lightweight Flask backend
* 📊 Uses structured data for accurate answers
* 🖥️ Simple, browser-based UI
* 🛠️ Ready for NLP/ML model upgrades
* 📱 Mobile-friendly interface

## Tech Stack

### Backend

* **Flask** – Python web framework
* **Python 3.8+** – Core language
* **NumPy** – Numerical computations
* **TensorFlow** – Machine learning model support
* **NLTK** – Natural language processing
* **DigitalCommerce360 Dataset** – Primary data source

### Frontend

* **HTML / CSS** – Flask templates
* **JavaScript** – Client-side interactivity

## Getting Started

### Prerequisites

* Python 3.8+
* pip (Python package manager)

### Installation

1. Clone the repository:

```
git clone https://github.com/InshaadMerchant/DigitalCommerce360-Chatbot.git
cd DigitalCommerce360-Chatbot
```

2. Install dependencies:

```
pip install numpy tensorflow nltk flask
```

3. Run the development server:

```
python app.py
```

4. Open [http://localhost:5000](http://localhost:5000) in your browser.

## Project Structure

```
DigitalCommerce360-Chatbot/
├── app.py               # Main Flask application
├── chatbot.py           # Chat logic and NLP integration
├── data/                # DigitalCommerce360 dataset
├── templates/           # HTML templates
│   └── index.html
├── static/              # CSS, JS, images
│   └── style.css
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

## Components

### app.py

The main Flask application:

* Handles routes
* Renders the chat UI
* Processes incoming messages

### chatbot.py

Core chatbot logic:

* Processes user queries
* Searches DigitalCommerce360 dataset
* Generates relevant responses

### templates/index.html

Frontend chat interface:

* Displays conversation
* Handles user input
* Shows bot replies

## Backend Integration

The chatbot can integrate with:

* **Custom-trained NLP models** for improved understanding
* **Databases** (e.g., SQLite, PostgreSQL) for storing chat history

### Example API Endpoint

```
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_response = get_bot_reply(user_message)
    return jsonify({"response": bot_response})
```

## Customization

### Styling

* Update `static/style.css` for design changes
* Modify `templates/index.html` for layout updates

### Features

* Add user authentication
* Save chat history
* Enable multi-language support
* Integrate with larger LLMs (OpenAI, Hugging Face, etc.)


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

