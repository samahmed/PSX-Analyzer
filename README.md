# PSX-Analyzer
A Python tool to analyze Pakistan Stock Exchange data with popular technical indicators and generate trading recommendations.

---

## Installation
Ensure you have Python 3 installed. Then install required packages:
pip install pandas requests beautifulsoup4 numpy python-dateutil pytz

---

## Usage
Run script with:
python PSX_Analyze.py SYMBOLS [MONTHS]

- `SYMBOLS`: One or more PSX stock symbols separated by commas (e.g., `MEBL` or `EFERT,MARI,MEBL`)
- `MONTHS` (optional): Number of past months of data to fetch (default: 3)

---

## Examples

python PSX_Analyze.py MEBL

python PSX_Analyze.py MEBL 2

python PSX_Analyze.py EFERT,MARI,MEBL,FABL

python PSX_Analyze.py EFERT,MARI,MEBL,FABL 2

---

## License

This project is provided as-is without any warranty. Please respect Pakistan Stock Exchange data usage rights and policies.



