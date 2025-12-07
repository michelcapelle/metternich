import os
import importlib.util
import numpy as np
import pandas as pd
import csv
from pathlib import Path

class KlemensHelper:

    def __init__(self, settings):
        self.settings = settings
        
    # https://academic.oup.com/isq/article-abstract/68/1/sqae001/7587491?redirectedFrom=fulltext
    # https://ucdp.uu.se/downloads/index.html#armedconflict
    # https://www.charliechaplin.com/en/synopsis/articles/29-The-Great-Dictator-s-Speech
    def load_debates(self):
        spec = importlib.util.spec_from_file_location("transform_UNGDC", "un-general-debates/data/transform_UNGDC.py")
        transform_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transform_module)
        original_dir = os.getcwd()
        os.chdir('un-general-debates/data')
        debates = transform_module.get_data()
        os.chdir(original_dir)
        debates = self.filter_countries(debates)
        if self.settings.is_benchmark:
            debates['text'] = "To those who can hear me, I say - do not despair. The misery that is now upon us is but the passing of greed - the bitterness of men who fear the way of human progress. The hate of men will pass, and dictators die, and the power they took from the people will return to the people. And so long as men die, liberty will never perish…"
        return debates

    def print_case_debates(self, debates):
        case_debates = debates[(debates['country'] == 'GBR') & (debates['year'] >= (str)(self.settings.case_from)) & (debates['year'] <= (str)(self.settings.case_to))].copy()
        print(f"Case debates:\n{case_debates}")

    def print_case_conflicts(self, conflicts):
        uk_code = str(200)
        case_conflicts = conflicts[
            (conflicts['start_date'] >= (str)(self.settings.case_from)) & 
            (conflicts['year'] >= self.settings.case_from) & 
            (conflicts['year'] <= self.settings.case_to + self.settings.max_window_years) & (
                (conflicts['gwno_a'].astype(str).str.contains(uk_code, na=False)) | 
                (conflicts['gwno_a_2nd'].astype(str).str.contains(uk_code, na=False)) |
                (conflicts['gwno_b'].astype(str).str.contains(uk_code, na=False)) |
                (conflicts['gwno_b_2nd'].astype(str).str.contains(uk_code, na=False)) |
                (conflicts['gwno_loc'].astype(str).str.contains(uk_code, na=False)))]
        columns_to_show = ['year', 'start_date', 'location', 'side_a', 'side_b', 'territory_name', 'intensity_level']
        print(f"Case conflicts: {case_conflicts[columns_to_show]}")

    # http://ksgleditsch.com/data-4.html
    def load_contries_codes(self):
        names = ['code', 'abbrev', 'country', 'start_date', 'end_date']
        ii_system = pd.read_csv('data/iisystem.dat', sep='\t', header=None, names=names, encoding='latin-1')
        drops = ['country', 'start_date', 'end_date']
        return ii_system.drop(drops, axis=1)

    def load_conflicts(self):
        conflicts = pd.read_csv('data/UcdpPrioConflict_v25_1.csv')
        conflicts = conflicts[conflicts['intensity_level'] == 2].copy()
        return conflicts
    
    # Top 25 ~ 71.6% of world population
    # https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population
    def filter_countries(self, data):
        top25 = ['IND', 'CHN', 'USA', 'IDN', 'PAK', 'NGA', 'BRA', 'BGD', 'RUS', 'MEX', 'JPN', 'PHL', 'COD', 'ETH', 'EGY', 'VNM', 'IRN', 'TUR', 'DEU', 'GBR', 'FRA', 'TZA', 'THA', 'ZAF', 'ITA']
        return data[data['country'].isin(top25)].copy()

    def write_results_to_csv(self, training_time, f1_score, war_prob_pred):
        if self.settings.do_test:
            return
        csv_path = Path("data/results.csv")
        fieldnames = ["run_id", "window_years", "is_benchmark", "war_pct", "training_time", "f1_score"]
        last_run_id = 0
        if csv_path.exists():
            with csv_path.open(newline="", mode="r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                rows = list(reader)
                if len(rows) > 1:
                    try:
                        last_run_id = int(rows[-1][0])
                    except (ValueError, IndexError):
                        last_run_id = 0
        new_run_id = last_run_id + 1
        is_benchmark = 1 if self.settings.is_benchmark else 0
        new_row = {
            "run_id": new_run_id,
            "window_years": self.settings.window_years,
            "is_benchmark": is_benchmark,
            "war_pct": (war_prob_pred * 100.0),
            "training_time": training_time,
            "f1_score": f1_score,
        }
        file_exists_and_not_empty = csv_path.exists() and csv_path.stat().st_size > 0
        with csv_path.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            if not file_exists_and_not_empty:
                writer.writeheader()
            writer.writerow(new_row)

    # https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
    def get_labels(self, debates, conflicts, countries):
        labels = []
        data_len = len(debates['country'])
        for i in range(0, data_len):
            country_abbrev = debates['country'].iloc[i]
            gwno_codes = countries[countries['abbrev'] == country_abbrev]['code'].values.tolist()      
            if not gwno_codes:
                if country_abbrev == 'URY':
                    gwno_codes = [165] # Uruguay
                elif country_abbrev == 'CZK' or country_abbrev == 'CSK':
                    gwno_codes = [315, 316, 317] # Czechoslovakia / Czech Republic / Slovakia
                elif country_abbrev == 'FRA':
                    gwno_codes = [220] # France
                elif country_abbrev == 'GBR':
                    gwno_codes = [200] # Great Britain
                elif country_abbrev == 'HTI':
                    gwno_codes = [41] # Haiti
                elif country_abbrev == 'LBN':
                    gwno_codes = [660] # Lebanon
                elif country_abbrev == 'NLD':
                    gwno_codes = [210] # Netherlands
                elif country_abbrev == 'NZL':
                    gwno_codes = [920] # New Zealand
                elif country_abbrev == 'PHL':
                    gwno_codes = [840] # Philippines
                elif country_abbrev == 'SWE':
                    gwno_codes = [380] # Sweden
                elif country_abbrev == 'THA':
                    gwno_codes = [800] # Thailand
                elif country_abbrev == 'TZA':
                    gwno_codes = [5101] # Tanzania
                elif country_abbrev == 'VAT' or country_abbrev == 'VCT':
                    gwno_codes = [327] # Papal State / Vatican City
                elif country_abbrev == 'VUT':
                    gwno_codes = [] # Vanuatu (404)
                elif country_abbrev == 'WSM':
                    gwno_codes = [] # Samoa (404)
                elif country_abbrev == 'ZAF':
                    gwno_codes = [560] # South Africa
                elif country_abbrev == 'ZMB':
                    gwno_codes = [551] # Zambia
                elif country_abbrev == 'ZWE':
                    gwno_codes = [552] # Zimbabwe
                elif country_abbrev == 'AGO':
                    gwno_codes = [540] # Angola
                elif country_abbrev == 'CPV':
                    gwno_codes = [402] # Cape Verde
                elif country_abbrev == 'CRI':
                    gwno_codes = [94] # Costa Rica
                elif country_abbrev == 'DEU':
                    gwno_codes = [260] # Germany et al.
                elif country_abbrev == 'DMA':
                    gwno_codes = [42] # Dominican Republic
                elif country_abbrev == 'DNK':
                    gwno_codes = [390] # Denmark
                elif country_abbrev == 'HRV':
                    gwno_codes = [344] # Croatia
                elif country_abbrev == 'IDN':
                    gwno_codes = [850] # Indonesia
                elif country_abbrev == 'IRL':
                    gwno_codes = [205] # Ireland
                elif country_abbrev == 'ISL':
                    gwno_codes = [395] # Iceland
                elif country_abbrev == 'LIE':
                    gwno_codes = [] # Liechtenstein (404)
                elif country_abbrev == 'LKA':
                    gwno_codes = [780] # Sri Lanka
                elif country_abbrev == 'LSO':
                    gwno_codes = [570] # Lesotho
                elif country_abbrev == 'LTU':
                    gwno_codes = [368] # Lithuania
                elif country_abbrev == 'LVA':
                    gwno_codes = [367] # Latvia
                elif country_abbrev == 'MAR':
                    gwno_codes = [600] # Morocco
                elif country_abbrev == 'MCO':
                    gwno_codes = [] # Monaco (404)
                elif country_abbrev == 'ROU':
                    gwno_codes = [] # Romania (404)
                elif country_abbrev == 'KAZ':
                    gwno_codes = [705] # Kazakhstan
                elif country_abbrev == 'KGZ':
                    gwno_codes = [703] # Kyrgych Republic
                elif country_abbrev == 'KHM':
                    gwno_codes = [811] # Cambodia
                elif country_abbrev == 'DZA':
                    gwno_codes = [615] # Algeria
                elif country_abbrev == 'ESP':
                    gwno_codes = [230] # Spain
                elif country_abbrev == 'EU':
                    gwno_codes = [] # European Union (404)
                elif country_abbrev == 'PRT':
                    gwno_codes = [235] # Portugal
                elif country_abbrev == 'PRY':
                    gwno_codes = [150] # Paraguay
                elif country_abbrev == 'PSE':
                    gwno_codes = [] # Palestine (404)
                elif country_abbrev == 'SDN':
                    gwno_codes = [625] # Sudan
                elif country_abbrev == 'MMR':
                    gwno_codes = [775] # Myanmar
                elif country_abbrev == 'GTM':
                    gwno_codes = [90] # Guatemala
                elif country_abbrev == 'HND':
                    gwno_codes = [91] # Honduras
                elif country_abbrev == 'BRB':
                    gwno_codes = [53] # Barbados
                elif country_abbrev == 'BTN':
                    gwno_codes = [760] # Bhutan
                elif country_abbrev == 'BWA':
                    gwno_codes = [571] # Botswana
                elif country_abbrev == 'CAF':
                    gwno_codes = [482] # Central African Republic
                elif country_abbrev == 'CHE':
                    gwno_codes = [225] # Switzerland
                elif country_abbrev == 'SRB':
                    gwno_codes = [340] # Serbia
                elif country_abbrev == 'STP':
                    gwno_codes = [] # Sao Tome and Principe (404)
                elif country_abbrev == 'SVK':
                    gwno_codes = [317] # Slovakia
                elif country_abbrev == 'SVN':
                    gwno_codes = [349] # Slovenia
                elif country_abbrev == 'SYC':
                    gwno_codes = [] # Seychelles (404)
                elif country_abbrev == 'AUT':
                    gwno_codes = [305] # Austria
                elif country_abbrev == 'BGR':
                    gwno_codes = [355] # Bulgaria
                elif country_abbrev == 'TTO':
                    gwno_codes = [52] # Trinidad and Tobago
                elif country_abbrev == 'YMD':
                    gwno_codes = [678, 680] # Yemen
                elif country_abbrev == 'COG':
                    gwno_codes = [484] # Congo
                elif country_abbrev == 'KWT':
                    gwno_codes = [690] # Kuwait
                elif country_abbrev == 'LBY':
                    gwno_codes = [620] # Libya
                elif country_abbrev == 'MDG':
                    gwno_codes = [580] # Madagascar
                elif country_abbrev == 'SGP':
                    gwno_codes = [830] # Singapore
                elif country_abbrev == 'NER':
                    gwno_codes = [436] # Niger
                elif country_abbrev == 'NGA':
                    gwno_codes = [475] # Nigeria
                elif country_abbrev == 'SLE':
                    gwno_codes = [451] # Sierra Leone
                elif country_abbrev == 'TCD':
                    gwno_codes = [483] # Chad
                elif country_abbrev == 'NPL':
                    gwno_codes = [790] # Nepal
                elif country_abbrev == 'MYS':
                    gwno_codes = [820] # Malaysia
                elif country_abbrev == 'COD':
                    gwno_codes = [490] # DR Congo
                elif country_abbrev == 'GEO':
                    gwno_codes = [372] # Georgia
                elif country_abbrev == 'GIN':
                    gwno_codes = [438] # Guinea
                elif country_abbrev == 'GMB':
                    gwno_codes = [420] # Gambia
                elif country_abbrev == 'GNQ':
                    gwno_codes = [411] # Equatorial Guinea
                elif country_abbrev == 'GRD':
                    gwno_codes = [] # Grenada (404)
                elif country_abbrev == 'KNA':
                    gwno_codes = [] # St. Kitts and Nevis (404)
                elif country_abbrev == 'LCA':
                    gwno_codes = [] # St. Lucia (404)
                elif country_abbrev == 'MDA':
                    gwno_codes = [359] # Moldova
                elif country_abbrev == 'MDV':
                    gwno_codes = [781] # Maldives
                elif country_abbrev == 'MHL':
                    gwno_codes = [] # Marshall Islands (404)
                elif country_abbrev == 'MKD':
                    gwno_codes = [343] # (North) Macedonia
                elif country_abbrev == 'MOZ':
                    gwno_codes = [541] # Mozambique
                elif country_abbrev == 'MRT':
                    gwno_codes = [435] # Mauritania
                elif country_abbrev == 'MUS':
                    gwno_codes = [590] # Mauritius
                elif country_abbrev == 'MWI':
                    gwno_codes = [553] # Malawi
                elif country_abbrev == 'NRU':
                    gwno_codes = [] # Nauru (404)
                elif country_abbrev == 'OMN':
                    gwno_codes = [698] # Oman
                elif country_abbrev == 'PLW':
                    gwno_codes = [] # Palau (404)
                elif country_abbrev == 'SLB':
                    gwno_codes = [940] # Solomon Islands
                elif country_abbrev == 'SMR':
                    gwno_codes = [] # San Marino (404)
                elif country_abbrev == 'TGO':
                    gwno_codes = [461] # Togo
                elif country_abbrev == 'TON':
                    gwno_codes = [] # Tonga (404)
                elif country_abbrev == 'TUV':
                    gwno_codes = [] # Tuvalu (404)
                elif country_abbrev == 'TJK':
                    gwno_codes = [702] # Tajikistan
                elif country_abbrev == 'CMR':
                    gwno_codes = [471] # Cameroon
                elif country_abbrev == 'AND':
                    gwno_codes = [] # Andorra (404)
                elif country_abbrev == 'ARE':
                    gwno_codes = [696] # United Arab Emirates
                elif country_abbrev == 'ATG':
                    gwno_codes = [] # Antigua and Barbuda (404)
                elif country_abbrev == 'BDI':
                    gwno_codes = [516] # Burundi
                elif country_abbrev == 'BFA':
                    gwno_codes = [439] # Burkina Faso
                elif country_abbrev == 'BGD':
                    gwno_codes = [771] # Bangladesh
                elif country_abbrev == 'BHR':
                    gwno_codes = [692] # Bahrain
                elif country_abbrev == 'BHS':
                    gwno_codes = [31] # Bahamas
                elif country_abbrev == 'BIH':
                    gwno_codes = [346] # Bosnia and Herzegovina
                elif country_abbrev == 'BRN':
                    gwno_codes = [835] # Brunei
                elif country_abbrev == 'CIV':
                    gwno_codes = [437] # Ivory Coast
                elif country_abbrev == 'FSM':
                    gwno_codes = [] # Micronesia (404)
                elif country_abbrev == 'KIR':
                    gwno_codes = [] # Kiribati (404)
                elif country_abbrev == 'MNE':
                    gwno_codes = [341] # Montenegro
                elif country_abbrev == 'TLS':
                    gwno_codes = [] # Timor-Leste (404)
                elif country_abbrev == 'DDR':
                    gwno_codes = [265] # East Germany
                else:
                    print(f"No GWNO code found for country: {country_abbrev}")
                    exit(1)
            year = debates['year'].iloc[i]
            if self.settings.window_years == 0:
                countries_future_war = conflicts[conflicts['year'] == year].drop('year', axis=1).values.flatten()
            elif self.settings.window_years > 0:
                countries_future_war = conflicts[(conflicts['year'] > year) & (conflicts['year'] <= year + self.settings.window_years)].drop('year', axis=1).values.flatten()
            else:
                countries_future_war = conflicts[(conflicts['year'] < year) & (conflicts['year'] >= year + self.settings.window_years)].drop('year', axis=1).values.flatten()
            countries_future_war = [x.split(', ') for x in countries_future_war if pd.notna(x)]
            countries_future_war = set([int(x) for sublist in countries_future_war for x in sublist])
            gwno_codes = set(gwno_codes)
            goes_to_war = 1.0 if gwno_codes & countries_future_war else 0.0        
            labels.append(goes_to_war)
        return np.array(labels)
