# App.py Documentation

## Overview

**app.py** is a Streamlit-based interactive web application that visualizes job posting data and associated skills across different geographical locations. The application provides two distinct viewing modes: a **Global Map** view showing all job postings, and a **Coverage Map** view showing primary/secondary city coverage areas with detailed skills breakdowns.

---

## Table of Contents

1. [Application Architecture](#application-architecture)
2. [Dependencies](#dependencies)
3. [Configuration](#configuration)
4. [Data Files](#data-files)
5. [Core Components](#core-components)
6. [Helper Functions](#helper-functions)
7. [Data Loading](#data-loading)
8. [Map Visualizations](#map-visualizations)
9. [User Interface](#user-interface)
10. [Session State Management](#session-state-management)
11. [Usage Guide](#usage-guide)
12. [Technical Details](#technical-details)

---

## Application Architecture

The application follows a modular architecture with clear separation of concerns:

```
app.py
├── Configuration & Styling (lines 16-131)
├── Helper Functions (lines 154-250)
├── Data Loading Functions (lines 266-569)
├── Data Processing & Caching (lines 574-650)
├── Map Visualization Functions (lines 879-1244)
└── Main Application Logic (lines 1247-1275)
```

---

## Dependencies

### Required Python Packages

Listed in [requirements.txt](requirements.txt):

- **streamlit**: Web application framework
- **pydeck**: WebGL-powered visualization library for large-scale data
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **python-calamine**: Fast Excel file reading (primary engine)
- **openpyxl**: Excel file reading (fallback engine)

### Standard Library Imports

- `re`: Regular expression operations
- `json`: JSON parsing for city configuration
- `hashlib`: Hashing functions (imported but not actively used)
- `io`: I/O operations for CSV downloads
- `pathlib`: Object-oriented filesystem paths
- `typing`: Type hints for better code documentation

---

## Configuration

### Page Configuration (lines 19-24)

```python
st.set_page_config(
    page_title="Job Posts & Skills Map",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
```

- **Wide layout** for better map visibility
- **Collapsed sidebar** for cleaner interface
- Earth emoji (🌍) as page icon

### Visual Constants (lines 136-150)

| Constant | Value | Purpose |
|----------|-------|---------|
| `UNKNOWN_SHEET_NAME` | "Unknown_Country" | Sheet name to filter out from data |
| `DEFAULT_COUNTRY_MIN_RADIUS` | 400m | Minimum bubble size for country view |
| `DEFAULT_COUNTRY_MAX_RADIUS` | 10,000m | Maximum bubble size for country view |
| `DEFAULT_GLOBAL_MIN_RADIUS_ALL` | 4,000m | Minimum bubble size for global view |
| `DEFAULT_GLOBAL_MAX_RADIUS_ALL` | 300,000m | Maximum bubble size for global view |
| `DEFAULT_FALLBACK_RADIUS_KM` | 10 km | Default radius when data is missing |

### Column Definitions (lines 155-171)

**Expected Skills Columns**:
- `Skill`: Name of the skill
- `SkillType`: Type classification (ST1=Hard, ST2=Soft)
- `City`: City name
- `Latitude`: Geographic latitude
- `Longitude`: Geographic longitude
- `count`: Frequency/occurrence count

**Column Aliases**: The application handles various column name variations through the `COL_ALIASES` dictionary, mapping common variations to standard names.

### Map Style (line 873)

Uses CartoDB's Voyager style without labels for a clean, modern appearance:
```python
MAP_STYLE = "https://basemaps.cartocdn.com/gl/voyager-nolabels-gl-style/style.json"
```

---

## Data Files

### Required Files

1. **primary_secondary_cities.json** (line 258)
   - Location: Project root directory
   - Format: JSON file containing city definitions
   - Required fields per city:
     - `country`: Country name
     - `City`/`city`: City name
     - `Latitude`/`latitude`: Geographic latitude
     - `Longitude`/`longitude`: Geographic longitude
     - `Radius`/`radius`: Coverage radius in kilometers
   - Purpose: Defines major cities and their coverage areas

2. **mapping/data/all_OJAs_aggregated.xlsx** (line 260)
   - Location: `mapping/data/` subdirectory
   - Format: Multi-sheet Excel workbook
   - Sheet structure: One sheet per country
   - Required columns per sheet:
     - `City`: City name
     - `Latitude`: Geographic latitude
     - `Longitude`: Geographic longitude
     - `Count`: Number of job postings
   - Purpose: Contains aggregated job posting counts by city

3. **mapping/data/cities_skills.xlsx** (line 259)
   - Location: `mapping/data/` subdirectory
   - Format: Multi-sheet Excel workbook
   - Sheet structure: One sheet per country
   - Required columns per sheet:
     - `Skill`: Skill name
     - `SkillType`: ST1 (Hard Skills) or ST2 (Soft Skills)
     - `City`: City name
     - `Latitude`: Geographic latitude
     - `Longitude`: Geographic longitude
     - `count`: Skill occurrence count
   - Purpose: Contains detailed skills data per city
   - Optional: Application continues without this file but with reduced functionality

---

## Core Components

### 1. CSS Styling (lines 27-131)

The application includes extensive custom CSS for a modern, polished appearance:

**Key Style Features**:
- Hidden Streamlit branding and sidebar
- Gradient-styled radio buttons with hover effects
- Modern data table styling with alternating row colors
- Compact selectbox styling
- Custom skills panel headers with gradients
- Responsive column layouts

**Color Scheme**:
- Primary gradient: Purple (#667eea to #764ba2)
- Success gradient: Teal (#11998e to #38ef7d)
- Accent colors for interactive elements

### 2. Data Normalization (lines 174-214)

**Text Normalization Functions**:

- `_canonicalize_columns(df)`: Maps column name variations to standard names
- `_normcase(x)`: Converts text to case-folded, whitespace-normalized form
- `_strip_weird_whitespace(s)`: Removes irregular whitespace
- `_clean_punctuation(s)`: Removes leading/trailing punctuation
- `_parse_int_series(s)`: Converts series to integers safely
- `_safe_to_numeric(s)`: Converts to numeric with error handling

### 3. Geographic Calculations (lines 224-234)

**Haversine Distance Formula**:

```python
def _haversine_km_vec(lat_arr, lon_arr, lat0, lon0) -> np.ndarray
```

Vectorized implementation calculating great-circle distances between points:
- **Input**: Arrays of latitudes/longitudes and a reference point
- **Output**: NumPy array of distances in kilometers
- **Earth Radius**: 6371.0088 km
- **Performance**: Optimized with NumPy for batch calculations

### 4. Visual Scaling (lines 216-221)

```python
def _scale_radius_by_global_sum(counts, global_sum, min_r, max_r)
```

Proportionally scales circle radii based on data values:
- Calculates proportion of each value relative to total
- Maps proportions to specified radius range
- Ensures visual differentiation while maintaining readability

---

## Data Loading

All data loading functions use `@st.cache_data` decorator for performance optimization.

### 1. Aggregated Job Data (lines 267-315)

**Function**: `load_aggregated_all_sheets(path, mtime)`

**Process**:
1. Attempts to use `calamine` engine (faster), falls back to `openpyxl`
2. Reads all sheets from the Excel workbook
3. For each sheet (country):
   - Normalizes column names
   - Validates required columns exist
   - Cleans city names and coordinates
   - Converts counts to integers
   - Creates normalized city names for matching
4. Filters out invalid entries (missing coordinates, empty city names)
5. Returns dictionary: `{country_name: DataFrame}`

**Output DataFrame Columns**:
- `Sheet`: Country name
- `City`: City display name
- `__City_norm__`: Normalized city name for matching
- `lat`: Latitude (float)
- `lon`: Longitude (float)
- `Count`: Job posting count (int)

### 2. Primary/Secondary Cities (lines 319-367)

**Function**: `load_primary_secondary_cities(path)`

**Process**:
1. Loads JSON file with city definitions
2. Handles both dictionary format `{"cities": [...]}` and direct array format
3. Normalizes column names (handles various naming conventions)
4. Validates required columns exist
5. Cleans and normalizes text fields
6. Converts coordinates to numeric values
7. Applies fallback radius where missing
8. Creates normalized fields for country and city matching

**Output DataFrame Columns**:
- `country`: Country name
- `Center_City`: City display name
- `Radius_km_max`: Coverage radius in kilometers
- `latitude`/`lat`: Latitude coordinate
- `longitude`/`lon`: Longitude coordinate
- `__City_norm__`: Normalized city name
- `__Country_norm__`: Normalized country name
- `Sheet`: Country name (for consistency)

### 3. Major Centers with Posts (lines 371-439)

**Function**: `build_major_centers_with_aggregated_posts(centers_df, aggregated_by_country_norm)`

**Purpose**: Combines city definitions with actual job posting data to calculate coverage statistics.

**Algorithm**:
1. Groups centers by country
2. For each center city:
   - Retrieves own job posting count
   - Calculates distances to all other cities in country using Haversine formula
   - Identifies cities within coverage radius
   - Sums job postings from nearby cities
   - Calculates total posts (own + included)
3. Only includes cities with at least one job posting

**Output DataFrame Columns**:
- `Sheet`: Country name
- `Center_City`: City name
- `__City_norm__`: Normalized city name
- `lat`/`lon`: Coordinates
- `Radius_km_max`: Coverage radius (km)
- `Own_posts`: Job posts directly in this city
- `Total_posts`: Own posts + posts from covered cities
- `Included_locations`: Count of nearby cities within radius

### 4. Skills Data (lines 492-568)

**Function**: `load_country_skills(path, mtime, country)`

**Process**:
1. Opens Excel file using cached handle
2. Parses specific country sheet
3. Canonicalizes column names
4. Cleans text fields (skills, skill types, cities)
5. Converts coordinates to numeric
6. Parses count as integers
7. Filters invalid entries
8. Creates normalized city field

**Function**: `build_city_geo_for_country(path, mtime, country)`

**Purpose**: Aggregates skills data to city level with weighted average coordinates.

**Algorithm**:
1. Loads all skills for country
2. Calculates weighted coordinates:
   - `weighted_lat = latitude × count`
   - `weighted_lon = longitude × count`
3. Groups by city and sums weights
4. Computes average: `lat = sum(weighted_lat) / sum(count)`
5. Falls back to simple mean if weighted average fails
6. Returns city-level aggregation

---

## Map Visualizations

### Version 1: Global Map (lines 879-1006)

**Purpose**: Displays all job posting locations as red circles scaled by posting volume.

**Features**:
- **Single dropdown**: Country selection (All or specific country)
- **Red circles**: Job posting locations
- **Size scaling**: Proportional to posting count
- **Hover tooltips**: Available only for priority cities
- **Text labels**: Display names for priority cities

**Technical Implementation**:

1. **Data Preparation**:
   - Aggregates data from all countries or selected country
   - Scales circle radii based on posting counts
   - Marks hoverable cities (priority cities only)

2. **Layers**:
   - Two `ScatterplotLayer` instances:
     - Non-hoverable circles (pickable=False) for regular cities
     - Hoverable circles (pickable=True) for priority cities
   - `TextLayer` for city labels (priority cities only)

3. **Color Scheme**:
   - Red circles: RGB(220, 50, 50) with alpha 180
   - Hover background: #c0392b

4. **Zoom Levels**:
   - All countries: zoom 3
   - Single country: zoom 5

**Priority Cities**: Defined by `ALWAYS_LABEL_CITIES` set, which includes:
- Cities manually specified in `ALWAYS_LABEL_CITIES_STATIC`
- Top posting city from each country (automatically determined)

### Version 2: Coverage Map (lines 1011-1244)

**Purpose**: Shows primary/secondary cities with coverage areas and detailed skills.

**Features**:
- **Two dropdowns**: Country and City selection
- **Green circles**: Coverage areas (radius = `Radius_km_max`)
- **Click interaction**: Click city to view skills
- **Skills panel**: Detailed breakdown when city selected
- **Adaptive zoom**: Adjusts based on city radius

**Technical Implementation**:

1. **Data Preparation**:
   - Loads major centers with coverage radii
   - Filters by country/city selection
   - Converts radius from km to meters for visualization
   - Highlights selected city with brighter green

2. **Layers**:
   - `ScatterplotLayer` with coverage circles
     - Radius: `Radius_km_max × 1000` (meters)
     - Pickable for click interaction
     - Stroked with dark green border
   - `TextLayer` for city labels
     - All cities shown when country selected
     - Only priority cities when viewing all countries

3. **Color Scheme**:
   - Default circles: RGB(34, 139, 34) with alpha 120
   - Selected city: RGB(0, 180, 0) with alpha 180
   - Hover background: #27ae60

4. **Adaptive Zoom** (lines 1124-1134):
   ```
   Radius ≤ 10 km  → zoom 10
   Radius ≤ 25 km  → zoom 9
   Radius ≤ 50 km  → zoom 8
   Radius ≤ 100 km → zoom 7
   Radius > 100 km → zoom 6
   ```

5. **Click Handling** (lines 827-859):
   - Extracts clicked city from PyDeck event
   - Updates session state with selection
   - Triggers rerun to update UI and load skills

### Skills Panel (lines 740-822)

**Purpose**: Displays top 50 hard and soft skills for selected city and coverage area.

**Features**:
- **Scope selection**: City only vs. coverage area (all cities within radius)
- **Dual columns**: Hard skills (ST1) and Soft skills (ST2)
- **Top 50 ranking**: Most frequent skills displayed
- **CSV download**: Export skill data for analysis
- **Modern styling**: Gradient headers, clean tables

**Algorithm**:

1. **Load country skills**: Retrieve all skills data for selected country
2. **Determine scope**:
   - If coverage included: Find all cities within radius using Haversine
   - If city only: Use selected city name
3. **Filter and aggregate**:
   - Filter skills by city scope
   - Group by skill name
   - Sum counts across locations
4. **Split by type**:
   - ST1 (Hard Skills): Technical, job-specific skills
   - ST2 (Soft Skills): Interpersonal, transferable skills
5. **Display top 50**: Sort by count descending, take top 50
6. **Generate download**: Create CSV with sanitized filename

**Filename Format**: `skills_{country}_{city}_{type}.csv`

---

## User Interface

### Main Layout (lines 1247-1275)

```
┌─────────────────────────────────────────┐
│ Job Posts & Skills Map                  │
├─────────────────────────────────────────┤
│ ○ Global Map                            │
│ ○ Coverage of Main Locations            │
├─────────────────────────────────────────┤
│ [View-specific description]             │
├─────────────────────────────────────────┤
│                                          │
│           [Map Visualization]            │
│                                          │
├─────────────────────────────────────────┤
│      [Skills Panel - if applicable]      │
└─────────────────────────────────────────┘
```

### Radio Button Selection (lines 1253-1258)

Two mutually exclusive views:
1. **"Global Map"**: Shows Version 1 visualization
2. **"Coverage of Main Locations"**: Shows Version 2 visualization

### Descriptive Captions

**Global Map** (lines 1263-1265):
> "Global view: red circles show job post counts for all cities; labels and hover tooltips are limited to priority cities. Hover over circles to see city details."

**Coverage Map** (lines 1269-1271):
> "Coverage view: green circles show coverage radii for all primary/secondary cities; labels show priority cities (or all cities when a country is selected). Click a city or select from dropdown to view skills (wait a moment for data to load)."

---

## Session State Management

### Version 1 State Variables

- `v1_country`: Selected country in Global Map view
  - Default: "All"
  - Options: "All" + list of available countries

### Version 2 State Variables

- `v2_country`: Selected country in Coverage Map view
  - Default: "All"
  - Options: "All" + list of countries with centers data

- `v2_city`: Selected city in Coverage Map view
  - Default: "All"
  - Options: Depends on country selection
  - Updates when country changes

### Click Interaction State (lines 848-868)

**Pending Selection Mechanism**:

1. User clicks city on map
2. `_extract_clicked_center(event)` extracts city/country from PyDeck event
3. `_set_dropdown_selection(country, city)` stores in temporary state:
   - `__pending_country`: Country to select
   - `__pending_city`: City to select
4. `st.rerun()` triggered
5. `_ingest_pending_selection()` applies pending values:
   - Sets `v2_country` to pending country
   - Stores city in `__pending_city_after_country__`
6. After country dropdown renders, city is applied to `v2_city`

**Why Two-Step Process?**

The country selection affects available cities. The process ensures:
1. Country dropdown updates first
2. City choices refresh based on new country
3. Then city selection is applied from valid options

---

## Usage Guide

### Initial Setup

1. **Prepare Data Files**:
   - Place `primary_secondary_cities.json` in project root
   - Create `mapping/data/` directory
   - Add `all_OJAs_aggregated.xlsx` to `mapping/data/`
   - (Optional) Add `cities_skills.xlsx` to `mapping/data/`

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application**:
   ```bash
   streamlit run app.py
   ```

### Navigation Guide

#### Viewing Global Distribution

1. Select **"Global Map"** radio button
2. Choose country from dropdown or keep "All" for global view
3. Hover over red circles to see city details
4. Observe relative posting volumes by circle size

#### Exploring City Coverage

1. Select **"Coverage of Main Locations"** radio button
2. Choose country from first dropdown
3. (Optional) Choose specific city from second dropdown
4. Green circles show coverage areas
5. Click any green circle to view that city's skills

#### Analyzing Skills

1. In Coverage view, select a specific city
2. Wait for skills panel to load below map
3. Review top 50 hard skills (left column)
4. Review top 50 soft skills (right column)
5. Click "Download CSV" to export data

### Performance Considerations

**Caching Strategy**:
- All data loading functions use `@st.cache_data`
- Cache keys include file modification time (`mtime`)
- Cache automatically invalidates when data files update
- Excel file handle uses `@st.cache_resource` for efficiency

**Load Times**:
- Initial load: Reads all Excel sheets (may take several seconds)
- View switching: Nearly instant (data cached)
- City selection: Skills load on demand (1-2 seconds)
- Country switching: Instant (data pre-loaded)

---

## Technical Details

### Excel Engine Selection (lines 237-242)

```python
def _best_excel_engine() -> str:
    try:
        import calamine
        return "calamine"
    except Exception:
        return "openpyxl"
```

**Rationale**:
- `calamine`: Rust-based, extremely fast for large files
- `openpyxl`: Pure Python, universal compatibility
- Auto-detection ensures best available engine is used

### Distance Calculation Details (lines 224-234)

**Haversine Formula Implementation**:

1. Convert coordinates to radians
2. Calculate differences: Δlat, Δlon
3. Apply formula:
   ```
   a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
   c = 2 × atan2(√a, √(1-a))
   distance = R × c
   ```
4. Earth radius R = 6371.0088 km (mean radius)

**Vectorization**: Uses NumPy operations for entire arrays, providing 10-100× speedup vs. loops.

### Data Validation Patterns

**Common Validation Steps**:

1. **Null/Empty Checks**:
   ```python
   df = df[df["City"].notna() & (df["City"] != "") & (df["City"] != "nan")]
   ```

2. **Coordinate Validation**:
   ```python
   df = df.dropna(subset=["lat", "lon"])
   ```

3. **Normalized Matching**:
   ```python
   df["__City_norm__"] = df["City"].apply(_normcase)
   df = df[~df["__City_norm__"].isin({"", "nan"})]
   ```

4. **Type Conversion with Error Handling**:
   ```python
   pd.to_numeric(series, errors="coerce")
   ```

### Color Coding System

**Global Map (Version 1)**:
- **Red** (#DC3232): Job posting locations
- Purpose: Draw attention to data density
- Opacity: 180/255 (70%) for overlapping visibility

**Coverage Map (Version 2)**:
- **Forest Green** (#228B22): Standard coverage areas
- **Bright Green** (#00B400): Selected city (highlighted)
- **Dark Green** (#006400): Circle borders
- Opacity varies: 80-180/255 based on selection state

**UI Elements**:
- **Purple Gradient**: Primary buttons and headers
- **Teal Gradient**: Active/selected states
- **Light backgrounds**: #f8f9fa for contrast

### Tooltip Configuration

**Global Map Tooltip**:
```html
<b>{City}</b><br/>Posts: {Count}
```
- Shows city name and posting count
- Red background (#c0392b) matching circle color

**Coverage Map Tooltip**:
```html
<b>{Center_City}</b><br/>Coverage: {Radius_km_max} km<br/>Total Posts: {Total_posts}
```
- Shows city name, coverage radius, and total posts
- Green background (#27ae60) matching circle color

### Computation Efficiency

**Global Totals** (lines 443-486):

Pre-computed at startup to avoid repeated calculations:
- `GLOBAL_TOTAL_POSTS`: Sum across all major centers
- `TOTAL_POSTS_BY_COUNTRY`: Dictionary of country totals
- `GLOBAL_TOTAL_AGGREGATED`: Sum across all aggregated data
- `COUNTRY_TOTAL_AGGREGATED`: Dictionary of aggregated totals per country

**Lazy Loading**:
- Skills data only loads when city selected
- Country skills parsed on-demand, not at startup
- Geographic aggregation cached per country

### Known Limitations

1. **File Dependencies**: Application stops with error if required files missing
2. **Sheet Names**: Must exactly match country names in city configuration
3. **Memory Usage**: Large Excel files fully loaded into memory
4. **Skills Panel**: Requires exact country/city match in skills workbook
5. **Label Overlap**: Text labels may overlap in dense areas
6. **Unknown Countries**: Sheet named "Unknown_Country" is filtered out

### Error Handling Strategy

**Fatal Errors** (using `st.error()` + `st.stop()`):
- Missing `primary_secondary_cities.json`
- Invalid JSON format in city configuration
- Missing required columns in data files
- Missing `all_OJAs_aggregated.xlsx`
- No valid country sheets found

**Graceful Degradation**:
- Skills file missing: Shows info message, continues without skills
- Empty data for country: Shows info message
- No skills for city: Shows info message
- Invalid/missing coordinates: Filtered out silently

---

## Code Quality Features

### Type Hints

Extensive use of type hints for clarity:
```python
def _haversine_km_vec(lat_arr, lon_arr, lat0, lon0) -> np.ndarray
def load_aggregated_all_sheets(path: Path, mtime: float) -> Dict[str, pd.DataFrame]
def _normcase(x: Optional[str]) -> str
```

### Naming Conventions

- **Private functions**: Prefixed with `_` (e.g., `_normcase`, `_clean_punctuation`)
- **Internal columns**: Double underscore wrapper (e.g., `__City_norm__`, `__Country_norm__`)
- **Constants**: ALL_CAPS (e.g., `GLOBAL_TOTAL_POSTS`, `MAP_STYLE`)
- **Session state keys**: Descriptive with version prefix (e.g., `v1_country`, `v2_city`)

### Documentation

- **Docstrings**: Functions include purpose descriptions
- **Inline comments**: Explain complex logic sections
- **Section headers**: ASCII art dividers organize code into logical sections

### Modularity

Functions are single-purpose and composable:
- Data loading separate from processing
- Visualization separate from data manipulation
- Helper functions reusable across contexts
- Clear input/output contracts

---

## Maintenance Notes

### Adding New Countries

1. Add country sheet to `all_OJAs_aggregated.xlsx`
2. Add city definitions to `primary_secondary_cities.json`
3. (Optional) Add country sheet to `cities_skills.xlsx`
4. Restart application to clear cache

### Modifying Visualization

**Change circle sizes**:
- Adjust constants: `DEFAULT_COUNTRY_MIN_RADIUS`, `DEFAULT_COUNTRY_MAX_RADIUS`
- Modify for global view: `DEFAULT_GLOBAL_MIN_RADIUS_ALL`, `DEFAULT_GLOBAL_MAX_RADIUS_ALL`

**Change colors**:
- Version 1 red circles: lines 933-936
- Version 2 green circles: lines 1099-1115
- Update tooltip backgrounds: lines 989, 1191

**Modify zoom levels**:
- Global view: line 914 (zoom=3)
- Country view: lines 925, 1080 (zoom=5)
- City-specific: lines 1124-1134

### Performance Tuning

**Reduce memory usage**:
- Load only required columns with `usecols` parameter
- Filter data earlier in processing pipeline
- Use chunked reading for extremely large files

**Improve speed**:
- Ensure `python-calamine` is installed
- Pre-compute more aggregations at startup
- Reduce number of layers in visualization
- Limit label rendering to priority cities only

---

## Conclusion

This application provides a comprehensive, interactive visualization platform for exploring job posting and skills data across geographical locations. Its dual-view architecture balances global overview with detailed local analysis, while efficient caching and modern UI design ensure a responsive user experience.

The modular codebase facilitates maintenance and extension, with clear separation between data loading, processing, and visualization concerns. Extensive validation and error handling ensure robustness across diverse data scenarios.
