# Job Posts & Skills Map App 🌍

An interactive Streamlit web application that visualizes job posting data and associated skills across different geographical locations.

## 🔗 Live Demo

**Deployed on Streamlit:** [https://cities-skills-v2.streamlit.app/](https://cities-skills-v2.streamlit.app/)

## 📋 About

This application provides comprehensive visualization of job market data with two main viewing modes:

- **Global Map View**: Displays all job postings across the world with interactive bubble visualization
- **Coverage Map View**: Shows primary/secondary city coverage areas with detailed skills breakdowns by city

### Key Features

- 🗺️ Interactive map visualization using PyDeck (WebGL-powered)
- 📊 Skills analysis and breakdown (Hard Skills vs Soft Skills)
- 🔍 Dynamic filtering by country and city
- 📈 Real-time data aggregation and statistics
- 💾 CSV export functionality for filtered data
- 🎨 Modern, responsive UI with gradient styling

### Data Includes

- Job posting locations (latitude/longitude coordinates)
- Skills categorization (ST1: Hard Skills, ST2: Soft Skills)
- Country and city-level aggregations
- Primary/secondary city coverage radius data

## 🚀 How to Run Locally

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "OJAs Map App"
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

   or if it didn't work:
   ```bash
   python -m streamlit run app.py
   ```

2. The app will automatically open in your default browser at `http://localhost:8501`

## 📦 Dependencies

- **streamlit** - Web application framework
- **pydeck** - WebGL-powered visualization library
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **python-calamine** - Fast Excel file reading
- **openpyxl** - Excel file reading (fallback)

See [requirements.txt](requirements.txt) for the complete list.

## 📂 Project Structure

```
OJAs Map App/
├── app.py                           # Main application
├── requirements.txt                 # Python dependencies
├── mapping/                         # Data files and configurations
│   └── primary_secondary_cities.json
├── skills_by_country_aggregated/    # Skills data by country (Excel files)
├── APP_DOCUMENTATION.md             # Detailed technical documentation
├── App Technical Documentation.pdf  # Technical documentation (PDF)
└── App_User_Guide.pdf              # User guide (PDF)
```

## 📖 Documentation

For detailed technical documentation, see:
- [APP_DOCUMENTATION.md](APP_DOCUMENTATION.md) - Complete technical documentation
- [App Technical Documentation.pdf](App%20Technical%20Documentation.pdf) - Technical documentation in PDF format
- [App_User_Guide.pdf](App_User_Guide.pdf) - User guide in PDF format

## 🛠️ Configuration

The app is pre-configured with default settings for visualization. Key configurations include:

- Bubble radius ranges for different zoom levels
- Color schemes for skill types
- Default fallback values for missing data

All configurations can be found in [app.py](app.py) lines 136-150.

## 🌐 Usage

1. **Select View Mode**: Choose between "Global Map" or "Coverage Map"
2. **Filter Data**: Select country and city from dropdown menus
3. **Explore Skills**: View skills breakdowns in interactive tables
4. **Export Data**: Download filtered data as CSV files
5. **Interact with Map**: Zoom, pan, and click on bubbles for details

## 📝 License

[Add your license information here]

## 👥 Contributors

[Add contributor information here]

## 🐛 Issues & Support

For issues or questions, please [open an issue](../../issues) in the repository.
