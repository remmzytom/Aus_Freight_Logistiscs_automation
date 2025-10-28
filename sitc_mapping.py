"""
SITC Code to Product Description Mapping
Standard International Trade Classification (SITC) mapping for Australian export data
"""

# Comprehensive SITC code to product description mapping
SITC_MAPPING = {
    # Section 0: Food and Live Animals
    '01': 'Live animals',
    '02': 'Meat and meat preparations',
    '03': 'Fish, crustaceans, molluscs and preparations thereof',
    '04': 'Cereals and cereal preparations',
    '05': 'Vegetables and fruit',
    '06': 'Sugars, sugar preparations and honey',
    '07': 'Coffee, tea, cocoa, spices, and manufactures thereof',
    '08': 'Feeding stuff for animals (not including unmilled cereals)',
    '09': 'Miscellaneous edible products and preparations',
    
    # Section 1: Beverages and Tobacco
    '11': 'Beverages',
    '12': 'Tobacco and tobacco manufactures',
    
    # Section 2: Crude Materials (Except Fuels)
    '21': 'Hides, skins and furskins, raw',
    '22': 'Oil-seeds and oleaginous fruits',
    '23': 'Crude rubber (including synthetic and reclaimed)',
    '24': 'Cork and wood',
    '25': 'Pulp and waste paper',
    '26': 'Textile fibres (not wool tops) and their wastes',
    '27': 'Crude fertilizers and crude minerals (excluding coal, petroleum and precious stones)',
    '28': 'Metalliferous ores and metal scrap',
    '29': 'Crude animal and vegetable materials, n.e.s.',
    
    # Section 3: Mineral Fuels and Lubricants
    '32': 'Coal, coke and briquettes',
    '33': 'Petroleum, petroleum products and related materials',
    '34': 'Gas, natural and manufactured',
    '35': 'Electric current',
    
    # Section 4: Animal and Vegetable Oils, Fats and Waxes
    '41': 'Animal oils and fats',
    '42': 'Fixed vegetable fats and oils, crude, refined or fractionated',
    '43': 'Animal or vegetable fats and oils, processed; waxes of animal or vegetable origin; inedible mixtures or preparations of animal or vegetable fats or oils',
    
    # Section 5: Chemicals and Related Products
    '51': 'Organic chemicals',
    '52': 'Inorganic chemicals',
    '53': 'Dyeing, tanning and colouring materials',
    '54': 'Medicinal and pharmaceutical products',
    '55': 'Essential oils and resinoids and perfume materials; toilet, polishing and cleaning preparations',
    '56': 'Fertilizers, manufactured',
    '57': 'Explosives and pyrotechnic products',
    '58': 'Plastics in non-primary forms',
    '59': 'Chemical materials and products, n.e.s.',
    
    # Section 6: Manufactured Goods Classified by Material
    '61': 'Leather, leather manufactures, n.e.s., and dressed furskins',
    '62': 'Rubber manufactures, n.e.s.',
    '63': 'Cork and wood manufactures (excluding furniture)',
    '64': 'Paper, paperboard and articles of paper pulp, of paper or of paperboard',
    '65': 'Textile yarn, fabrics, made-up articles, n.e.s., and related products',
    '66': 'Non-metallic mineral manufactures, n.e.s.',
    '67': 'Iron and steel',
    '68': 'Non-ferrous metals',
    '69': 'Manufactures of metal, n.e.s.',
    
    # Section 7: Machinery and Transport Equipment
    '71': 'Power-generating machinery and equipment',
    '72': 'Machinery specialized for particular industries',
    '73': 'Metalworking machinery',
    '74': 'General industrial machinery and equipment, n.e.s., and machine parts, n.e.s.',
    '75': 'Office machines and automatic data-processing machines',
    '76': 'Telecommunications and sound-recording and reproducing apparatus and equipment',
    '77': 'Electrical machinery, apparatus and appliances, n.e.s., and electrical parts thereof',
    '78': 'Road vehicles (including air-cushion vehicles)',
    '79': 'Other transport equipment',
    
    # Section 8: Miscellaneous Manufactured Articles
    '81': 'Prefabricated buildings; sanitary, plumbing, heating and lighting fixtures and fittings, n.e.s.',
    '82': 'Furniture and parts thereof; bedding, mattresses, mattress supports, cushions and similar stuffed furnishings',
    '83': 'Travel goods, handbags and similar containers',
    '84': 'Articles of apparel and clothing accessories',
    '85': 'Footwear',
    '87': 'Professional, scientific and controlling instruments and apparatus, n.e.s.',
    '88': 'Photographic apparatus, equipment and supplies and optical goods, n.e.s.; watches and clocks',
    '89': 'Miscellaneous manufactured articles, n.e.s.',
    
    # Section 9: Commodities and Transactions Not Classified Elsewhere
    '91': 'Postal packages not classified according to kind',
    '93': 'Special transactions and commodities not classified according to kind',
    '96': 'Coin (other than gold coin), not being legal tender',
    '97': 'Gold, non-monetary (excluding gold ores and concentrates)',
    '98': 'Special transactions and commodities not classified according to kind',
    '99': 'Commodities and transactions not classified elsewhere in the SITC'
}


def map_sitc_to_product(sitc_code):
    """
    Map SITC code to product description
    
    Args:
        sitc_code: SITC code (string or numeric)
        
    Returns:
        str: Product description based on SITC code
    """
    import pandas as pd
    
    if pd.isna(sitc_code) or sitc_code == '':
        return 'Unknown Product'
    
    sitc_str = str(sitc_code).strip()
    
    # Handle 5-digit SITC codes by taking first 2 digits
    if len(sitc_str) >= 2:
        section_code = sitc_str[:2]
        return SITC_MAPPING.get(section_code, f'Product Code {sitc_code}')
    
    return f'Product Code {sitc_code}'


def get_sitc_sections():
    """
    Get list of all SITC sections with descriptions
    
    Returns:
        dict: SITC sections with codes and descriptions
    """
    return SITC_MAPPING.copy()


def get_unclassified_patterns():
    """
    Get patterns that indicate unclassified products
    
    Returns:
        list: List of patterns to match for unclassified products
    """
    return ['unclassified', 'confidential', 'not included', 'miscellaneous', 'other', 'product code']


if __name__ == "__main__":
    # Test the mapping function
    test_codes = ['28011', '33000', '67000', '75000', '99999']
    print("SITC Code Mapping Test:")
    for code in test_codes:
        result = map_sitc_to_product(code)
        print(f"  {code} → {result}")
