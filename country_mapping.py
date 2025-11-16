"""
Country Code to Country Name Mapping
ISO 2-letter country code mapping for Australian export data
"""

# Comprehensive country code to country name mapping (ISO 2-letter codes)
COUNTRY_MAPPING = {
    'AD': 'Andorra', 'AE': 'United Arab Emirates', 'AF': 'Afghanistan', 'AG': 'Antigua and Barbuda',
    'AI': 'Anguilla', 'AL': 'Albania', 'AM': 'Armenia', 'AO': 'Angola', 'AQ': 'Antarctica',
    'AR': 'Argentina', 'AS': 'American Samoa', 'AT': 'Austria', 'AU': 'Australia',
    'AW': 'Aruba', 'AX': 'Åland Islands', 'AZ': 'Azerbaijan', 'BA': 'Bosnia and Herzegovina',
    'BB': 'Barbados', 'BD': 'Bangladesh', 'BE': 'Belgium', 'BF': 'Burkina Faso',
    'BG': 'Bulgaria', 'BH': 'Bahrain', 'BI': 'Burundi', 'BJ': 'Benin',
    'BL': 'Saint Barthélemy', 'BM': 'Bermuda', 'BN': 'Brunei Darussalam', 'BO': 'Bolivia',
    'BQ': 'Bonaire, Sint Eustatius and Saba', 'BR': 'Brazil', 'BS': 'Bahamas', 'BT': 'Bhutan',
    'BV': 'Bouvet Island', 'BW': 'Botswana', 'BY': 'Belarus', 'BZ': 'Belize',
    'CA': 'Canada', 'CC': 'Cocos (Keeling) Islands', 'CD': 'Congo, Democratic Republic of the',
    'CF': 'Central African Republic', 'CG': 'Congo', 'CH': 'Switzerland', 'CI': 'Côte d\'Ivoire',
    'CK': 'Cook Islands', 'CL': 'Chile', 'CM': 'Cameroon', 'CN': 'China', 'CHIN': 'China',
    'CO': 'Colombia', 'CR': 'Costa Rica', 'CU': 'Cuba', 'CV': 'Cape Verde',
    'CW': 'Curaçao', 'CX': 'Christmas Island', 'CY': 'Cyprus', 'CZ': 'Czech Republic',
    'DE': 'Germany', 'DJ': 'Djibouti', 'DK': 'Denmark', 'DM': 'Dominica',
    'DO': 'Dominican Republic', 'DZ': 'Algeria', 'EC': 'Ecuador', 'EE': 'Estonia',
    'EG': 'Egypt', 'EH': 'Western Sahara', 'ER': 'Eritrea', 'ES': 'Spain',
    'ET': 'Ethiopia', 'FI': 'Finland', 'FJ': 'Fiji', 'FK': 'Falkland Islands (Malvinas)',
    'FM': 'Micronesia, Federated States of', 'FO': 'Faroe Islands', 'FR': 'France',
    'GA': 'Gabon', 'GB': 'United Kingdom', 'GD': 'Grenada', 'GE': 'Georgia',
    'GF': 'French Guiana', 'GG': 'Guernsey', 'GH': 'Ghana', 'GI': 'Gibraltar',
    'GL': 'Greenland', 'GM': 'Gambia', 'GN': 'Guinea', 'GP': 'Guadeloupe',
    'GQ': 'Equatorial Guinea', 'GR': 'Greece', 'GS': 'South Georgia and the South Sandwich Islands',
    'GT': 'Guatemala', 'GU': 'Guam', 'GW': 'Guinea-Bissau', 'GY': 'Guyana',
    'HK': 'Hong Kong', 'HM': 'Heard Island and McDonald Islands', 'HN': 'Honduras',
    'HR': 'Croatia', 'HT': 'Haiti', 'HU': 'Hungary', 'ID': 'Indonesia',
    'IE': 'Ireland', 'IL': 'Israel', 'IM': 'Isle of Man', 'IN': 'India',
    'IO': 'British Indian Ocean Territory', 'IQ': 'Iraq', 'IR': 'Iran, Islamic Republic of',
    'IS': 'Iceland', 'IT': 'Italy', 'JE': 'Jersey', 'JM': 'Jamaica',
    'JO': 'Jordan', 'JP': 'Japan', 'KE': 'Kenya', 'KG': 'Kyrgyzstan',
    'KH': 'Cambodia', 'KI': 'Kiribati', 'KM': 'Comoros', 'KN': 'Saint Kitts and Nevis',
    'KP': 'Korea, Democratic People\'s Republic of', 'KR': 'Korea, Republic of', 'KW': 'Kuwait',
    'KY': 'Cayman Islands', 'KZ': 'Kazakhstan', 'LA': 'Lao People\'s Democratic Republic',
    'LB': 'Lebanon', 'LC': 'Saint Lucia', 'LI': 'Liechtenstein', 'LK': 'Sri Lanka',
    'LR': 'Liberia', 'LS': 'Lesotho', 'LT': 'Lithuania', 'LU': 'Luxembourg',
    'LV': 'Latvia', 'LY': 'Libya', 'MA': 'Morocco', 'MC': 'Monaco',
    'MD': 'Moldova, Republic of', 'ME': 'Montenegro', 'MF': 'Saint Martin (French part)',
    'MG': 'Madagascar', 'MH': 'Marshall Islands', 'MK': 'North Macedonia',
    'ML': 'Mali', 'MM': 'Myanmar', 'MN': 'Mongolia', 'MO': 'Macao',
    'MP': 'Northern Mariana Islands', 'MQ': 'Martinique', 'MR': 'Mauritania',
    'MS': 'Montserrat', 'MT': 'Malta', 'MU': 'Mauritius', 'MV': 'Maldives',
    'MW': 'Malawi', 'MX': 'Mexico', 'MY': 'Malaysia', 'MZ': 'Mozambique',
    'NA': 'Namibia', 'NC': 'New Caledonia', 'NE': 'Niger', 'NF': 'Norfolk Island',
    'NG': 'Nigeria', 'NI': 'Nicaragua', 'NL': 'Netherlands', 'NO': 'Norway',
    'NP': 'Nepal', 'NR': 'Nauru', 'NU': 'Niue', 'NZ': 'New Zealand',
    'OM': 'Oman', 'PA': 'Panama', 'PE': 'Peru', 'PF': 'French Polynesia',
    'PG': 'Papua New Guinea', 'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland',
    'PM': 'Saint Pierre and Miquelon', 'PN': 'Pitcairn', 'PR': 'Puerto Rico',
    'PS': 'Palestine, State of', 'PT': 'Portugal', 'PW': 'Palau', 'PY': 'Paraguay',
    'QA': 'Qatar', 'RE': 'Réunion', 'RO': 'Romania', 'RS': 'Serbia',
    'RU': 'Russian Federation', 'RW': 'Rwanda', 'SA': 'Saudi Arabia', 'SB': 'Solomon Islands',
    'SC': 'Seychelles', 'SD': 'Sudan', 'SE': 'Sweden', 'SG': 'Singapore',
    'SH': 'Saint Helena, Ascension and Tristan da Cunha', 'SI': 'Slovenia',
    'SJ': 'Svalbard and Jan Mayen', 'SK': 'Slovakia', 'SL': 'Sierra Leone',
    'SM': 'San Marino', 'SN': 'Senegal', 'SO': 'Somalia', 'SR': 'Suriname',
    'SS': 'South Sudan', 'ST': 'Sao Tome and Principe', 'SV': 'El Salvador',
    'SX': 'Sint Maarten (Dutch part)', 'SY': 'Syrian Arab Republic', 'SZ': 'Eswatini',
    'TC': 'Turks and Caicos Islands', 'TD': 'Chad', 'TF': 'French Southern Territories',
    'TG': 'Togo', 'TH': 'Thailand', 'TJ': 'Tajikistan', 'TK': 'Tokelau',
    'TL': 'Timor-Leste', 'TM': 'Turkmenistan', 'TN': 'Tunisia', 'TO': 'Tonga',
    'TR': 'Turkey', 'TT': 'Trinidad and Tobago', 'TV': 'Tuvalu', 'TW': 'Taiwan',
    'TZ': 'Tanzania, United Republic of', 'UA': 'Ukraine', 'UG': 'Uganda',
    'UM': 'United States Minor Outlying Islands', 'US': 'United States of America',
    'UY': 'Uruguay', 'UZ': 'Uzbekistan', 'VA': 'Holy See', 'VC': 'Saint Vincent and the Grenadines',
    'VE': 'Venezuela, Bolivarian Republic of', 'VG': 'Virgin Islands, British',
    'VI': 'Virgin Islands, U.S.', 'VN': 'Viet Nam', 'VU': 'Vanuatu',
    'WF': 'Wallis and Futuna', 'WS': 'Samoa', 'YE': 'Yemen', 'YT': 'Mayotte',
    'ZA': 'South Africa', 'ZM': 'Zambia', 'ZW': 'Zimbabwe',
    # Common abbreviations and special cases
    'NCD': 'Confidential / Not Published',  # ABS confidential destination
    '999999': 'Confidential / Not Published',  # ABS confidential code
    '999': 'Confidential / Not Published',  # ABS confidential code
    'CHIN': 'China',  # China abbreviation
    'VIE': 'Viet Nam',  # Vietnam abbreviation
    'FRAN': 'France',  # France abbreviation
    'CAN': 'Canada',  # Canada abbreviation
    'HONG': 'Hong Kong',  # Hong Kong abbreviation
    'VIET': 'Vietnam',  # Vietnam abbreviation
    'GHAN': 'Ghana',  # Ghana abbreviation
    'NAUR': 'Nauru',  # Nauru abbreviation
    'BOTS': 'Botswana',  # Botswana abbreviation
    'UNKN': 'Unknown',  # Unknown abbreviation
    'BURK': 'Burkina Faso',  # Burkina Faso abbreviation
    'MACA': 'Macau',  # Macau abbreviation
    'ARGE': 'Argentina',  # Argentina abbreviation
    'NAMI': 'Namibia',  # Namibia abbreviation
    'PNMA': 'Panama',  # Panama abbreviation
    'VENE': 'Venezuela',  # Venezuela abbreviation
    'TUNI': 'Tunisia',  # Tunisia abbreviation
    'MALI': 'Mali',  # Mali abbreviation
    'NIGE': 'Niger',  # Niger abbreviation
    'CHAD': 'Chad',  # Chad abbreviation
    'CAMR': 'Cameroon',  # Cameroon abbreviation
    'GABO': 'Gabon',  # Gabon abbreviation
    'CONG': 'Congo',  # Congo abbreviation
    'RWAN': 'Rwanda',  # Rwanda abbreviation
    'BURU': 'Burundi',  # Burundi abbreviation
    'DJIB': 'Djibouti',  # Djibouti abbreviation
    'ERIT': 'Eritrea',  # Eritrea abbreviation
    'SOMAL': 'Somalia',  # Somalia abbreviation
    'SUDAN': 'Sudan',  # Sudan abbreviation
    'LIBY': 'Libya',  # Libya abbreviation
    'WEST': 'Western Sahara',  # Western Sahara abbreviation
    'CAPE': 'Cape Verde',  # Cape Verde abbreviation
    'GUIN': 'Guinea',  # Guinea abbreviation
    'SIER': 'Sierra Leone',  # Sierra Leone abbreviation
    'LIBE': 'Liberia',  # Liberia abbreviation
    'IVOR': 'Ivory Coast',  # Ivory Coast abbreviation
    'GAMB': 'Gambia',  # Gambia abbreviation
    'BENI': 'Benin',  # Benin abbreviation
    'TOGO': 'Togo',  # Togo abbreviation
    'LESO': 'Lesotho',  # Lesotho abbreviation
    'SWAZ': 'Swaziland',  # Swaziland abbreviation
    'MALA': 'Malawi',  # Malawi abbreviation
    'COMO': 'Comoros',  # Comoros abbreviation
    'SEYC': 'Seychelles',  # Seychelles abbreviation
    'SRNM': 'Suriname',  # Suriname abbreviation
    'WALL': 'Wallis and Futuna',  # Wallis and Futuna abbreviation
    'MRNS': 'Morocco',  # Morocco abbreviation
    'GUYA': 'Guyana',  # Guyana abbreviation
    'BOHR': 'Bahrain',  # Bahrain abbreviation
    'STVI': 'Saint Vincent and the Grenadines',  # Saint Vincent abbreviation
    'GNDA': 'Grenada',  # Grenada abbreviation
    'IWAS': 'International Waters',  # International Waters abbreviation
    'STHE': 'Saint Helena'  # Saint Helena abbreviation
}


def map_country_code_to_name(country_code):
    """
    Map country code to country name
    
    Args:
        country_code: ISO 2-letter country code (string)
        
    Returns:
        str: Full country name based on country code
    """
    import pandas as pd
    
    if pd.isna(country_code) or country_code == '':
        return 'Unknown Country'
    
    country_str = str(country_code).strip().upper()
    
    # Try exact match first
    if country_str in COUNTRY_MAPPING:
        return COUNTRY_MAPPING[country_str]
    
    # Handle ABS confidential codes (often numeric)
    if country_str in {'999999', '999'}:
        return 'Confidential / Not Published'
    
    # If not found, return a more meaningful name instead of "Country Code XXX"
    # This prevents creating "Country Code XXX" entries that cause regional mapping issues
    return 'Unknown Country'


def get_country_mapping():
    """
    Get complete country mapping dictionary
    
    Returns:
        dict: Country code to country name mapping
    """
    return COUNTRY_MAPPING.copy()


def get_problematic_country_patterns():
    """
    Get patterns that indicate missing or problematic country entries
    
    Returns:
        list: List of patterns to match for problematic countries
    """
    return [
        'no country details', 'unknown', 'n/a', 'na', 'null', 'missing', 
        'not specified', 'other', 'miscellaneous', 'tbd', 'pending'
    ]

def is_problematic_country_name(country_name):
    """
    Check if a country name is problematic and needs mapping
    
    Args:
        country_name: Country name to check
        
    Returns:
        bool: True if country name is problematic, False otherwise
    """
    import pandas as pd
    
    if pd.isna(country_name) or country_name == '':
        return True
        
    country_lower = str(country_name).strip().lower()
    problematic_patterns = get_problematic_country_patterns()
    
    # Check if it exactly matches a problematic pattern
    for pattern in problematic_patterns:
        if pattern in country_lower:
            return True
    
    return False


def validate_country_code(country_code):
    """
    Validate if a country code exists in the mapping
    
    Args:
        country_code: ISO 2-letter country code (string)
        
    Returns:
        bool: True if country code exists, False otherwise
    """
    if pd.isna(country_code) or country_code == '':
        return False
    
    country_str = str(country_code).strip().upper()
    return country_str in COUNTRY_MAPPING


if __name__ == "__main__":
    # Test the mapping function
    test_codes = ['AU', 'US', 'CN', 'JP', 'DE', 'GB', 'INVALID']
    print("Country Code Mapping Test:")
    for code in test_codes:
        result = map_country_code_to_name(code)
        valid = validate_country_code(code)
        print(f"  {code} → {result} (Valid: {valid})")
