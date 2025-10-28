"""
Regional Mapping for Australian Export Data Analysis
Country-to-Region classification for strategic market analysis
"""

# Comprehensive regional mapping for Australian export destinations
REGION_MAPPING = {
    # Asia-Pacific
    'China': 'Asia-Pacific',
    'Japan': 'Asia-Pacific', 
    'South Korea': 'Asia-Pacific',
    'India': 'Asia-Pacific',
    'Singapore': 'Asia-Pacific',
    'Hong Kong': 'Asia-Pacific',
    'Taiwan': 'Asia-Pacific',
    'Thailand': 'Asia-Pacific',
    'Malaysia': 'Asia-Pacific',
    'Indonesia': 'Asia-Pacific',
    'Philippines': 'Asia-Pacific',
    'Vietnam': 'Asia-Pacific',
    'Viet Nam': 'Asia-Pacific',
    'Bangladesh': 'Asia-Pacific',
    'Pakistan': 'Asia-Pacific',
    'Sri Lanka': 'Asia-Pacific',
    'Myanmar': 'Asia-Pacific',
    'Cambodia': 'Asia-Pacific',
    'Laos': 'Asia-Pacific',
    'Brunei': 'Asia-Pacific',
    'Papua New Guinea': 'Asia-Pacific',
    'Fiji': 'Asia-Pacific',
    'New Zealand': 'Asia-Pacific',
    'Mongolia': 'Asia-Pacific',
    'Nepal': 'Asia-Pacific',
    'Bhutan': 'Asia-Pacific',
    'Maldives': 'Asia-Pacific',
    'Solomon Islands': 'Asia-Pacific',
    'Vanuatu': 'Asia-Pacific',
    'Samoa': 'Asia-Pacific',
    'Tonga': 'Asia-Pacific',
    'Kiribati': 'Asia-Pacific',
    'Tuvalu': 'Asia-Pacific',
    'Nauru': 'Asia-Pacific',
    'Palau': 'Asia-Pacific',
    'Marshall Islands': 'Asia-Pacific',
    'Micronesia': 'Asia-Pacific',
    
    # North America
    'United States': 'North America',
    'Canada': 'North America',
    'Mexico': 'North America',
    'Guatemala': 'North America',
    'Belize': 'North America',
    'El Salvador': 'North America',
    'Honduras': 'North America',
    'Nicaragua': 'North America',
    'Costa Rica': 'North America',
    'Panama': 'North America',
    
    # Europe
    'United Kingdom': 'Europe',
    'Germany': 'Europe',
    'France': 'Europe',
    'Netherlands': 'Europe',
    'Italy': 'Europe',
    'Spain': 'Europe',
    'Belgium': 'Europe',
    'Switzerland': 'Europe',
    'Austria': 'Europe',
    'Sweden': 'Europe',
    'Norway': 'Europe',
    'Denmark': 'Europe',
    'Finland': 'Europe',
    'Poland': 'Europe',
    'Russia': 'Europe',
    'Turkey': 'Europe',
    'Greece': 'Europe',
    'Portugal': 'Europe',
    'Ireland': 'Europe',
    'Czech Republic': 'Europe',
    'Hungary': 'Europe',
    'Romania': 'Europe',
    'Bulgaria': 'Europe',
    'Croatia': 'Europe',
    'Slovenia': 'Europe',
    'Slovakia': 'Europe',
    'Estonia': 'Europe',
    'Latvia': 'Europe',
    'Lithuania': 'Europe',
    'Luxembourg': 'Europe',
    'Malta': 'Europe',
    'Cyprus': 'Europe',
    'Iceland': 'Europe',
    'Liechtenstein': 'Europe',
    'Monaco': 'Europe',
    'San Marino': 'Europe',
    'Vatican City': 'Europe',
    'Andorra': 'Europe',
    'Ukraine': 'Europe',
    'Belarus': 'Europe',
    'Moldova': 'Europe',
    'Serbia': 'Europe',
    'Montenegro': 'Europe',
    'North Macedonia': 'Europe',
    'Albania': 'Europe',
    'Bosnia and Herzegovina': 'Europe',
    'Kosovo': 'Europe',
    
    # Middle East & Africa
    'Saudi Arabia': 'Middle East & Africa',
    'United Arab Emirates': 'Middle East & Africa',
    'Israel': 'Middle East & Africa',
    'South Africa': 'Middle East & Africa',
    'Egypt': 'Middle East & Africa',
    'Morocco': 'Middle East & Africa',
    'Nigeria': 'Middle East & Africa',
    'Kenya': 'Middle East & Africa',
    'Ghana': 'Middle East & Africa',
    'Algeria': 'Middle East & Africa',
    'Tunisia': 'Middle East & Africa',
    'Jordan': 'Middle East & Africa',
    'Lebanon': 'Middle East & Africa',
    'Kuwait': 'Middle East & Africa',
    'Qatar': 'Middle East & Africa',
    'Bahrain': 'Middle East & Africa',
    'Oman': 'Middle East & Africa',
    'Iran': 'Middle East & Africa',
    'Iraq': 'Middle East & Africa',
    'Yemen': 'Middle East & Africa',
    'Syria': 'Middle East & Africa',
    'Afghanistan': 'Middle East & Africa',
    'Ethiopia': 'Middle East & Africa',
    'Uganda': 'Middle East & Africa',
    'Tanzania': 'Middle East & Africa',
    'Zimbabwe': 'Middle East & Africa',
    'Zambia': 'Middle East & Africa',
    'Botswana': 'Middle East & Africa',
    'Namibia': 'Middle East & Africa',
    'Angola': 'Middle East & Africa',
    'Mozambique': 'Middle East & Africa',
    'Madagascar': 'Middle East & Africa',
    'Mauritius': 'Middle East & Africa',
    'Senegal': 'Middle East & Africa',
    'Mali': 'Middle East & Africa',
    'Burkina Faso': 'Middle East & Africa',
    'Niger': 'Middle East & Africa',
    'Chad': 'Middle East & Africa',
    'Cameroon': 'Middle East & Africa',
    'Central African Republic': 'Middle East & Africa',
    'Democratic Republic of the Congo': 'Middle East & Africa',
    'Republic of the Congo': 'Middle East & Africa',
    'Gabon': 'Middle East & Africa',
    'Equatorial Guinea': 'Middle East & Africa',
    'São Tomé and Príncipe': 'Middle East & Africa',
    'Rwanda': 'Middle East & Africa',
    'Burundi': 'Middle East & Africa',
    'Djibouti': 'Middle East & Africa',
    'Eritrea': 'Middle East & Africa',
    'Somalia': 'Middle East & Africa',
    'Sudan': 'Middle East & Africa',
    'South Sudan': 'Middle East & Africa',
    'Libya': 'Middle East & Africa',
    'Tunisia': 'Middle East & Africa',
    'Western Sahara': 'Middle East & Africa',
    'Cape Verde': 'Middle East & Africa',
    'Guinea-Bissau': 'Middle East & Africa',
    'Guinea': 'Middle East & Africa',
    'Sierra Leone': 'Middle East & Africa',
    'Liberia': 'Middle East & Africa',
    'Ivory Coast': 'Middle East & Africa',
    'Gambia': 'Middle East & Africa',
    'Benin': 'Middle East & Africa',
    'Togo': 'Middle East & Africa',
    'Lesotho': 'Middle East & Africa',
    'Swaziland': 'Middle East & Africa',
    'Malawi': 'Middle East & Africa',
    'Comoros': 'Middle East & Africa',
    'Seychelles': 'Middle East & Africa',
    
    # South America
    'Brazil': 'South America',
    'Argentina': 'South America',
    'Chile': 'South America',
    'Colombia': 'South America',
    'Peru': 'South America',
    'Venezuela': 'South America',
    'Ecuador': 'South America',
    'Uruguay': 'South America',
    'Paraguay': 'South America',
    'Bolivia': 'South America',
    'Guyana': 'South America',
    'Suriname': 'South America',
    'French Guiana': 'South America',
    
    # Central America & Caribbean
    'Cuba': 'Central America & Caribbean',
    'Jamaica': 'Central America & Caribbean',
    'Trinidad and Tobago': 'Central America & Caribbean',
    'Barbados': 'Central America & Caribbean',
    'Dominican Republic': 'Central America & Caribbean',
    'Haiti': 'Central America & Caribbean',
    'Bahamas': 'Central America & Caribbean',
    'Dominica': 'Central America & Caribbean',
    'Grenada': 'Central America & Caribbean',
    'Saint Kitts and Nevis': 'Central America & Caribbean',
    'Saint Lucia': 'Central America & Caribbean',
    'Saint Vincent and the Grenadines': 'Central America & Caribbean',
    'Antigua and Barbuda': 'Central America & Caribbean',
    'Belize': 'Central America & Caribbean',
    
    # Additional European countries
    'Cyprus': 'Europe',
    'Iceland': 'Europe',
    'Liechtenstein': 'Europe',
    'Monaco': 'Europe',
    'San Marino': 'Europe',
    'Vatican City': 'Europe',
    'Andorra': 'Europe',
    'Ukraine': 'Europe',
    'Belarus': 'Europe',
    'Moldova': 'Europe',
    'Serbia': 'Europe',
    'Montenegro': 'Europe',
    'North Macedonia': 'Europe',
    'Albania': 'Europe',
    'Bosnia and Herzegovina': 'Europe',
    'Kosovo': 'Europe',
    
    # Additional Asian countries
    'Mongolia': 'Asia-Pacific',
    'Nepal': 'Asia-Pacific',
    'Bhutan': 'Asia-Pacific',
    'Maldives': 'Asia-Pacific',
    'Kazakhstan': 'Asia-Pacific',
    'Kyrgyzstan': 'Asia-Pacific',
    'Tajikistan': 'Asia-Pacific',
    'Turkmenistan': 'Asia-Pacific',
    'Uzbekistan': 'Asia-Pacific',
    'Afghanistan': 'Asia-Pacific',
    
    # Additional African countries
    'Ethiopia': 'Middle East & Africa',
    'Uganda': 'Middle East & Africa',
    'Tanzania': 'Middle East & Africa',
    'Zimbabwe': 'Middle East & Africa',
    'Zambia': 'Middle East & Africa',
    'Botswana': 'Middle East & Africa',
    'Namibia': 'Middle East & Africa',
    'Angola': 'Middle East & Africa',
    'Mozambique': 'Middle East & Africa',
    'Madagascar': 'Middle East & Africa',
    'Mauritius': 'Middle East & Africa',
    'Senegal': 'Middle East & Africa',
    'Mali': 'Middle East & Africa',
    'Burkina Faso': 'Middle East & Africa',
    'Niger': 'Middle East & Africa',
    'Chad': 'Middle East & Africa',
    'Cameroon': 'Middle East & Africa',
    'Central African Republic': 'Middle East & Africa',
    'Democratic Republic of the Congo': 'Middle East & Africa',
    'Republic of the Congo': 'Middle East & Africa',
    'Gabon': 'Middle East & Africa',
    'Equatorial Guinea': 'Middle East & Africa',
    'São Tomé and Príncipe': 'Middle East & Africa',
    'Rwanda': 'Middle East & Africa',
    'Burundi': 'Middle East & Africa',
    'Djibouti': 'Middle East & Africa',
    'Eritrea': 'Middle East & Africa',
    'Somalia': 'Middle East & Africa',
    'Sudan': 'Middle East & Africa',
    'South Sudan': 'Middle East & Africa',
    'Libya': 'Middle East & Africa',
    'Western Sahara': 'Middle East & Africa',
    'Cape Verde': 'Middle East & Africa',
    'Guinea-Bissau': 'Middle East & Africa',
    'Guinea': 'Middle East & Africa',
    'Sierra Leone': 'Middle East & Africa',
    'Liberia': 'Middle East & Africa',
    'Ivory Coast': 'Middle East & Africa',
    'Gambia': 'Middle East & Africa',
    'Benin': 'Middle East & Africa',
    'Togo': 'Middle East & Africa',
    'Lesotho': 'Middle East & Africa',
    'Swaziland': 'Middle East & Africa',
    'Malawi': 'Middle East & Africa',
    'Comoros': 'Middle East & Africa',
    'Seychelles': 'Middle East & Africa',
    
    # Additional South American countries
    'Guyana': 'South America',
    'Suriname': 'South America',
    'French Guiana': 'South America',
    
    # Additional Central American countries
    'Guatemala': 'Central America & Caribbean',
    'Belize': 'Central America & Caribbean',
    'El Salvador': 'Central America & Caribbean',
    'Honduras': 'Central America & Caribbean',
    'Nicaragua': 'Central America & Caribbean',
    'Costa Rica': 'Central America & Caribbean',
    'Panama': 'Central America & Caribbean',
    'Cuba': 'Central America & Caribbean',
    'Jamaica': 'Central America & Caribbean',
    'Trinidad and Tobago': 'Central America & Caribbean',
    'Barbados': 'Central America & Caribbean',
    'Dominican Republic': 'Central America & Caribbean',
    'Haiti': 'Central America & Caribbean',
    'Bahamas': 'Central America & Caribbean',
    'Dominica': 'Central America & Caribbean',
    'Grenada': 'Central America & Caribbean',
    'Saint Kitts and Nevis': 'Central America & Caribbean',
    'Saint Lucia': 'Central America & Caribbean',
    'Saint Vincent and the Grenadines': 'Central America & Caribbean',
    'Antigua and Barbuda': 'Central America & Caribbean',
    
    # Fix country name mismatches from actual data
    'China (excludes SARs and Taiwan)': 'Asia-Pacific',
    'Korea, Republic of (South)': 'Asia-Pacific',
    'Hong Kong (SAR of China)': 'Asia-Pacific',
    'Macau (SAR of China)': 'Asia-Pacific',
    'Brunei Darussalam': 'Asia-Pacific',
    'Congo, Democratic Republic of': 'Middle East & Africa',
    'Congo, Republic of': 'Middle East & Africa',
    'Cote d\'Ivoire': 'Middle East & Africa',
    'Czechia': 'Europe',
    'Denmark (includes Greenland and Faroe Islands)': 'Europe',
    'France (includes Andorra and Monaco)': 'Europe',
    'Italy (includes Holy See and San Marino)': 'Europe',
    'Armenia': 'Europe',
    'Azerbaijan': 'Europe',
    'Georgia': 'Europe',
    'Eswatini': 'Middle East & Africa',
    'Cabo Verde': 'Middle East & Africa',
    
    # Territories and special cases
    'Bermuda': 'North America',
    'Cayman Islands': 'Central America & Caribbean',
    'French Antilles (Guadeloupe and Martinique)': 'Central America & Caribbean',
    'French Polynesia': 'Asia-Pacific',
    'Cook Islands': 'Asia-Pacific',
    'Gibraltar': 'Europe',
    'Guam': 'Asia-Pacific',
    'Christmas Island': 'Asia-Pacific',
    'Cocos (Keeling) Islands': 'Asia-Pacific',
    'Falkland Islands (includes South Georgia and South Sandwich Islands)': 'South America',
    'International Waters': 'Other',
    'Antarctica, nfd': 'Other',
    'Australia (Re-imports)': 'Other',
    
    # Additional country name variations
    'Russian Federation': 'Europe',
    'Mauritania': 'Middle East & Africa',
    'Micronesia, Federated States of': 'Asia-Pacific',
    'Netherlands Antilles, nfd': 'Central America & Caribbean',
    'New Caledonia': 'Asia-Pacific',
    'Niue': 'Asia-Pacific',
    'No Country Details': 'Other',
    'Norfolk Island': 'Asia-Pacific',
    'Northern Mariana Islands': 'Asia-Pacific',
    'Pitcairn Islands': 'Asia-Pacific',
    'Puerto Rico': 'North America',
    'Reunion': 'Middle East & Africa',
    'Samoa, American': 'Asia-Pacific',
    'Ship and Aircraft Stores': 'Other',
    'Southern and East Africa, nec': 'Middle East & Africa',
    'St Helena': 'Middle East & Africa',
    'St Kitts and Nevis': 'Central America & Caribbean',
    'St Lucia': 'Central America & Caribbean',
    'St Pierre and Miquelon': 'North America',
    'St Vincent and the Grenadines': 'Central America & Caribbean',
    
    # Final country name variations
    'Switzerland (includes Liechtenstein)': 'Europe',
    'Timor-Leste': 'Asia-Pacific',
    'Turkiye': 'Europe',
    'Turks and Caicos Islands': 'Central America & Caribbean',
    'United Kingdom, Channel Islands and Isle of Man, nfd': 'Europe',
    'United States of America': 'North America',
    'Unknown': 'Other',
    'Virgin Islands, British': 'Central America & Caribbean',
    'Virgin Islands, United States': 'Central America & Caribbean',
    'Wallis and Futuna': 'Asia-Pacific',
    
    # Country codes from cleaned data
    'Country Code HONG': 'Asia-Pacific',  # Hong Kong
    'Country Code CAN': 'North America',  # Canada
    'Country Code VIET': 'Asia-Pacific',  # Vietnam
    'Country Code FRAN': 'Europe',        # France
    'Country Code GHAN': 'Middle East & Africa',  # Ghana
    'Country Code NAUR': 'Asia-Pacific',  # Nauru
    'Country Code BOTS': 'Middle East & Africa',  # Botswana
    'Country Code UNKN': 'Other',         # Unknown
    'Country Code BURK': 'Middle East & Africa',  # Burkina Faso
    'Country Code MACA': 'Asia-Pacific',  # Macau
    'Country Code ARGE': 'South America', # Argentina
    'Country Code NAMI': 'Middle East & Africa',  # Namibia
    'Country Code PNMA': 'Central America & Caribbean',  # Panama
    'Country Code VENE': 'South America', # Venezuela
    'Country Code TUNI': 'Middle East & Africa',  # Tunisia
    'Country Code MALI': 'Middle East & Africa',  # Mali
    'Country Code NIGE': 'Middle East & Africa',  # Niger
    'Country Code CHAD': 'Middle East & Africa',  # Chad
    'Country Code CAMR': 'Middle East & Africa',  # Cameroon
    'Country Code GABO': 'Middle East & Africa',  # Gabon
    'Country Code CONG': 'Middle East & Africa',  # Congo
    'Country Code RWAN': 'Middle East & Africa',  # Rwanda
    'Country Code BURU': 'Middle East & Africa',  # Burundi
    'Country Code DJIB': 'Middle East & Africa',  # Djibouti
    'Country Code ERIT': 'Middle East & Africa',  # Eritrea
    'Country Code SOMAL': 'Middle East & Africa', # Somalia
    'Country Code SUDAN': 'Middle East & Africa', # Sudan
    'Country Code LIBY': 'Middle East & Africa',  # Libya
    'Country Code WEST': 'Middle East & Africa',  # Western Sahara
    'Country Code CAPE': 'Middle East & Africa',  # Cape Verde
    'Country Code GUIN': 'Middle East & Africa',  # Guinea
    'Country Code SIER': 'Middle East & Africa',  # Sierra Leone
    'Country Code LIBE': 'Middle East & Africa',  # Liberia
    'Country Code IVOR': 'Middle East & Africa',  # Ivory Coast
    'Country Code GAMB': 'Middle East & Africa',  # Gambia
    'Country Code BENI': 'Middle East & Africa',  # Benin
    'Country Code TOGO': 'Middle East & Africa',  # Togo
    'Country Code LESO': 'Middle East & Africa',  # Lesotho
    'Country Code SWAZ': 'Middle East & Africa',  # Swaziland
    'Country Code MALA': 'Middle East & Africa',  # Malawi
    'Country Code COMO': 'Middle East & Africa',  # Comoros
    'Country Code SEYC': 'Middle East & Africa',  # Seychelles
    'Country Code SRNM': 'South America',  # Suriname
    'Country Code WALL': 'Asia-Pacific',   # Wallis and Futuna
    'Country Code MRNS': 'Middle East & Africa',  # Morocco
    'Country Code GUYA': 'South America',  # Guyana
    'Country Code BOHR': 'Middle East & Africa',  # Bahrain
    'Country Code STVI': 'Central America & Caribbean',  # Saint Vincent
    'Country Code GNDA': 'Central America & Caribbean',  # Grenada
    'Country Code IWAS': 'Other',          # International Waters
    'Country Code STHE': 'Middle East & Africa',  # Saint Helena
    
    # Additional country name variations found in cleaned data
    'Macau': 'Asia-Pacific',  # Macau (simplified name)
    'Saint Helena': 'Middle East & Africa',  # Saint Helena
    
    # Default for truly unmapped countries
    'Other': 'Other'
}

def map_country_to_region(country_name):
    """
    Map a country name to its region.
    
    Args:
        country_name (str): Name of the country
        
    Returns:
        str: Region name, or 'Other' if not found
    """
    return REGION_MAPPING.get(country_name, 'Other')

def get_region_statistics():
    """
    Get statistics about the regional mapping.
    
    Returns:
        dict: Statistics about the mapping
    """
    regions = set(REGION_MAPPING.values())
    countries_per_region = {}
    
    for country, region in REGION_MAPPING.items():
        if region not in countries_per_region:
            countries_per_region[region] = 0
        countries_per_region[region] += 1
    
    return {
        'total_countries': len(REGION_MAPPING),
        'total_regions': len(regions),
        'regions': sorted(regions),
        'countries_per_region': countries_per_region
    }

def add_region_to_dataframe(df, country_column='country_of_destination', region_column='region', debug_other=False):
    """
    Add region mapping to a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing country data
        country_column (str): Name of the column containing country names
        region_column (str): Name of the column to add for regions
        debug_other (bool): If True, print countries mapped to 'Other'
        
    Returns:
        pandas.DataFrame: DataFrame with region column added
    """
    df_copy = df.copy()
    df_copy[region_column] = df_copy[country_column].map(REGION_MAPPING).fillna('Other')
    
    if debug_other:
        other_countries = df_copy[df_copy[region_column] == 'Other'][country_column].unique()
        if len(other_countries) > 0:
            print(f"\n Countries mapped to 'Other' ({len(other_countries)} countries):")
            for country in sorted(other_countries):
                print(f"   • {country}")
            print(f"\n Add these to REGION_MAPPING to improve regional analysis")
    
    return df_copy

if __name__ == "__main__":
    # Print mapping statistics when run directly
    stats = get_region_statistics()
    print("Regional Mapping Statistics:")
    print(f"Total Countries Mapped: {stats['total_countries']}")
    print(f"Total Regions: {stats['total_regions']}")
    print("\nCountries per Region:")
    for region, count in stats['countries_per_region'].items():
        print(f"  {region}: {count} countries")
