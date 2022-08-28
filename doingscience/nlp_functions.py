import html
import json
import logging
import re
from urllib.parse import unquote, urlparse

import requests
from doingscience.general_functions import isNaN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STOP_WORDS = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they',
              'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or',
              'who',
              'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
              'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she',
              'all',
              'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that',
              'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
              'only',
              'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
              'further', 'was', 'here', 'than']


def website_regexp_validator(url):
    if url:
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return re.match(regex, url) is not None


def get_digit(text):
    try:
        text = text.replace(',', '')
        return float(re.findall(r'\d+(?:\.)?(?:\d+)?', text)[0])
    except Exception as exc:
        return None


def website_validator(url):
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        if response.status_code <= 200:
            return url
        else:
            logger.info(response.status_code)
            return None
    except Exception as exc:
        logger.info(exc)
        if website_regexp_validator(url):
            return url
        else:
            return None


def escape_html_text(text):
    """escape strings for display in HTML"""
    rs = html.escape(text).replace(u'\n', u'<br />').replace(u'\t', u'&emsp;').replace(u'  ', u' &nbsp;')
    return rs if rs else ' '


def remove_special_symbols(str):
    """
    Remove special symbols
    :param x:
    """
    pattern = r"[^\w\s]+"
    return re.sub(pattern, "", str.lower(), flags=re.IGNORECASE)


def remove_special_symbol_chars(str, sep=""):
    pattern = r"[\u00E2\u20AC×<>;:_¿§«»ω⊙¤°℃℉€¥£¢¡®©™`~!@#$%^&\*\(\)\|\+\–\-\=\?\'\’\",\.\{\}\[\]\\\/]+"
    return re.sub(pattern, sep, str, flags=re.IGNORECASE)


def match_case(x, pattern, return_result=False, failcase=False):
    """
    Match pattern
    :param x:
    :param pattern:
    :param failcase:
    :param return_result:
    :return:
    """

    if not x:
        return False

    m_case = re.search(pattern, x, re.IGNORECASE)

    if not m_case:
        result = failcase
    else:
        if return_result:
            try:
                result = re.search(pattern, x, re.IGNORECASE).groups(0)[0]
            except Exception as exc:
                logger.error('match_case : ' + str(exc))
                result = failcase
        else:
            result = True
    return result


def match_all_case(x, pattern, failcase=[]):
    """
    Match pattern
    :param x:
    :param pattern:
    :param failcase:
    :return:
    """

    if not x:
        return False

    m_case = re.findall(pattern, x, re.IGNORECASE)

    if not m_case:
        result = failcase
    else:
        result = m_case
    return result


def clear_input_symbols(x, ret=None):
    """
    Clear inpute symbols
    :param x:
    :return:
    """
    if not x or isNaN(x):
        return ret

    try:
        rs = re.sub(r'''\n|\r|\\r|\\n''', '', x)
    except Exception as exc:
        logger.error('clear_input_symbols : ' + str(exc))
        rs = ret
    finally:
        return rs


def clean_url_link(url, no_domain=True):
    """
    Clean url link
    :param url:
    :param no_domain:
    :return:
    """
    if (not url) | (url in ['-', '--', '', ' ', ' --', '(not set)', 'undefined', '(undefined)']) | (
            not isinstance(url, str)):
        url = ''
        return url

    url = decode_uri(url)
    if no_domain:
        url = remove_domain(url)

    return url


def remove_elem(input, words):
    return list(filter(lambda x: x not in words, input))


def remove_domain(x):
    """
    Remove domain
    :param x:
    :return:
    """
    try:
        result = re.sub(r'(^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?(?:[^:\/?\n]+))', '', x, flags=re.IGNORECASE)
    except Exception as exc:
        logger.error('remove_domain : ' + str(exc))
        result = ''
    return result


def extract_email_domain(email):
    if (not email) | (email in ['-', '--', '', ' ', ' --']) | (not isinstance(email, str)):
        domain = ''
        return domain

    try:
        domain = re.search(r'@(.+)', email, flags=re.IGNORECASE).groups(0)[0]
    except Exception as exc:
        # logger.error('extract_email_domain : ' + str(exc))
        domain = ''
    return domain


def extract_email_subdomain(email):
    if (not email) | (email in ['-', '--', '', ' ', ' --']) | (not isinstance(email, str)):
        subdomain = ''
        return subdomain

    try:
        subdomain = re.search(r"@(.+)\.[\w]+$", email, flags=re.IGNORECASE).groups(0)[0]

    except Exception as exc:
        # logger.error('extract_email_subdomain : ' + str(exc))
        subdomain = ''
    return subdomain


def extract_domain(url):
    """
    Extract domain
    :param url:
    :return:
    """
    if (not url) | (url in ['-', '--', '', ' ', ' --']) | (not isinstance(url, str)):
        domain = ''
        return domain

    try:
        domain = re.search(r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?((?:www\.)?([^:\/?\n]+))', url,
                           flags=re.IGNORECASE).groups(0)[0]
    except Exception as exc:
        # logger.error('extract_domain : ' + str(exc))
        domain = ''
    return domain


def find_country_from_domain(d, return_code=True):
    domain = extract_domain(d)
    l = {'ac': 'Ascension Island (United Kingdom)',
         'ad': 'Andorra',
         'ae': 'United Arab Emirates',
         'af': 'Afghanistan',
         'ag': 'Antigua and Barbuda',
         'ai': 'Anguilla (United Kingdom)',
         'al': 'Albania',
         'am': 'Armenia',
         'ao': 'Angola',
         'aq': 'Antarctica',
         'ar': 'Argentina',
         'as': 'American Samoa (United States)',
         'at': 'Austria',
         'au': 'Australia',
         'aw': 'Aruba (Kingdom of the Netherlands)',
         'ax': 'Aland (Finland)',
         'az': 'Azerbaijan',
         'ba': 'Bosnia and Herzegovina',
         'bb': 'Barbados',
         'bd': 'Bangladesh',
         'be': 'Belgium',
         'bf': 'Burkina Faso',
         'bg': 'Bulgaria',
         'bh': 'Bahrain',
         'bi': 'Burundi',
         'bj': 'Benin',
         'bm': 'Bermuda (United Kingdom)',
         'bn': 'Brunei',
         'bo': 'Bolivia',
         'bq': 'Caribbean Netherlands ( Bonaire,  Saba, and  Sint Eustatius)',
         'br': 'Brazil',
         'bs': 'Bahamas',
         'bt': 'Bhutan',
         'bw': 'Botswana',
         'by': 'Belarus',
         'bz': 'Belize',
         'ca': 'Canada',
         'cc': 'Cocos (Keeling) Islands (Australia)',
         'cd': 'Democratic Republic of the Congo',
         'cf': 'Central African Republic',
         'cg': 'Republic of the Congo',
         'ch': 'Switzerland',
         'ci': 'Ivory Coast',
         'ck': 'Cook Islands',
         'cl': 'Chile',
         'cm': 'Cameroon',
         'cn': "People's Republic of China",
         'co': 'Colombia',
         'cr': 'Costa Rica',
         'cu': 'Cuba',
         'cv': 'Cape Verde',
         'cw': 'Curaçao (Kingdom of the Netherlands)',
         'cx': 'Christmas Island',
         'cy': 'Cyprus',
         'cz': 'Czech Republic',
         'de': 'Germany',
         'dj': 'Djibouti',
         'dk': 'Denmark',
         'dm': 'Dominica',
         'do': 'Dominican Republic',
         'dz': 'Algeria',
         'ec': 'Ecuador',
         'ee': 'Estonia',
         'eg': 'Egypt',
         'eh': 'Western Sahara',
         'er': 'Eritrea',
         'es': 'Spain',
         'et': 'Ethiopia',
         'eu': 'European Union',
         'fi': 'Finland',
         'fj': 'Fiji',
         'fk': 'Falkland Islands (United Kingdom)',
         'fm': 'Federated States of Micronesia',
         'fo': 'Faroe Islands (Kingdom of Denmark)',
         'fr': 'France',
         'ga': 'Gabon',
         'gd': 'Grenada',
         'ge': 'Georgia',
         'gf': 'French Guiana (France)',
         'gg': 'Guernsey (United Kingdom)',
         'gh': 'Ghana',
         'gi': 'Gibraltar (United Kingdom)',
         'gl': 'Greenland (Kingdom of Denmark)',
         'gm': 'The Gambia',
         'gn': 'Guinea',
         'gp': 'Guadeloupe (France)',
         'gq': 'Equatorial Guinea',
         'gr': 'Greece',
         'gs': 'South Georgia and the South Sandwich Islands (United Kingdom)',
         'gt': 'Guatemala',
         'gu': 'Guam (United States)',
         'gw': 'Guinea-Bissau',
         'gy': 'Guyana',
         'hk': 'Hong Kong',
         'hm': 'Heard Island and McDonald Islands',
         'hn': 'Honduras',
         'hr': 'Croatia',
         'ht': 'Haiti',
         'hu': 'Hungary',
         'id': 'Indonesia',
         'ie': 'Ireland',
         'il': 'Israel',
         'im': 'Isle of Man (United Kingdom)',
         'in': 'India',
         'io': 'British Indian Ocean Territory (United Kingdom)',
         'iq': 'Iraq',
         'ir': 'Iran',
         'is': 'Iceland',
         'it': 'Italy',
         'je': 'Jersey (United Kingdom)',
         'jm': 'Jamaica',
         'jo': 'Jordan',
         'jp': 'Japan',
         'ke': 'Kenya',
         'kg': 'Kyrgyzstan',
         'kh': 'Cambodia',
         'ki': 'Kiribati',
         'km': 'Comoros',
         'kn': 'Saint Kitts and Nevis',
         'kp': 'North Korea',
         'kr': 'South Korea',
         'kw': 'Kuwait',
         'ky': 'Cayman Islands (United Kingdom)',
         'kz': 'Kazakhstan',
         'la': 'Laos',
         'lb': 'Lebanon',
         'lc': 'Saint Lucia',
         'li': 'Liechtenstein',
         'lk': 'Sri Lanka',
         'lr': 'Liberia',
         'ls': 'Lesotho',
         'lt': 'Lithuania',
         'lu': 'Luxembourg',
         'lv': 'Latvia',
         'ly': 'Libya',
         'ma': 'Morocco',
         'mc': 'Monaco',
         'md': 'Moldova',
         'me': 'Montenegro',
         'mg': 'Madagascar',
         'mh': 'Marshall Islands',
         'mk': 'North Macedonia',
         'ml': 'Mali',
         'mm': 'Myanmar',
         'mn': 'Mongolia',
         'mo': 'Macau',
         'mp': 'Northern Mariana Islands (United States)',
         'mq': 'Martinique (France)',
         'mr': 'Mauritania',
         'ms': 'Montserrat (United Kingdom)',
         'mt': 'Malta',
         'mu': 'Mauritius',
         'mv': 'Maldives',
         'mw': 'Malawi',
         'mx': 'Mexico',
         'my': 'Malaysia',
         'mz': 'Mozambique',
         'na': 'Namibia',
         'nc': 'New Caledonia (France)',
         'ne': 'Niger',
         'nf': 'Norfolk Island',
         'ng': 'Nigeria',
         'ni': 'Nicaragua',
         'nl': 'Netherlands',
         'no': 'Norway',
         'np': 'Nepal',
         'nr': 'Nauru',
         'nu': 'Niue',
         'nz': 'New Zealand',
         'om': 'Oman',
         'pa': 'Panama',
         'pe': 'Peru',
         'pf': 'French Polynesia (France)',
         'pg': 'Papua New Guinea',
         'ph': 'Philippines',
         'pk': 'Pakistan',
         'pl': 'Poland',
         'pm': 'Saint-Pierre and Miquelon (France)',
         'pn': 'Pitcairn Islands (United Kingdom)',
         'pr': 'Puerto Rico (United States)',
         'ps': 'Palestine[56]',
         'pt': 'Portugal',
         'pw': 'Palau',
         'py': 'Paraguay',
         'qa': 'Qatar',
         're': 'Réunion (France)',
         'ro': 'Romania',
         'rs': 'Serbia',
         'ru': 'Russia',
         'rw': 'Rwanda',
         'sa': 'Saudi Arabia',
         'sb': 'Solomon Islands',
         'sc': 'Seychelles',
         'sd': 'Sudan',
         'se': 'Sweden',
         'sg': 'Singapore',
         'sh': 'Saint Helena, Ascension and Tristan da Cunha (United Kingdom)',
         'si': 'Slovenia',
         'sk': 'Slovakia',
         'sl': 'Sierra Leone',
         'sm': 'San Marino',
         'sn': 'Senegal',
         'so': 'Somalia',
         'sr': 'Suriname',
         'ss': 'South Sudan',
         'st': 'São Tomé and Príncipe',
         'su': 'Soviet Union',
         'sv': 'El Salvador',
         'sx': 'Sint Maarten (Kingdom of the Netherlands)',
         'sy': 'Syria',
         'sz': 'Eswatini',
         'tc': 'Turks and Caicos Islands (United Kingdom)',
         'td': 'Chad',
         'tf': 'French Southern and Antarctic Lands',
         'tg': 'Togo',
         'th': 'Thailand',
         'tj': 'Tajikistan',
         'tk': 'Tokelau',
         'tl': 'East Timor',
         'tm': 'Turkmenistan',
         'tn': 'Tunisia',
         'to': 'Tonga',
         'tr': 'Turkey',
         'tt': 'Trinidad and Tobago',
         'tv': 'Tuvalu',
         'tw': 'Taiwan',
         'tz': 'Tanzania',
         'ua': 'Ukraine',
         'ug': 'Uganda',
         'uk': 'United Kingdom',
         'us': 'United States of America',
         'com': 'United States of America',
         'uy': 'Uruguay',
         'uz': 'Uzbekistan',
         'va': 'Vatican City',
         'vc': 'Saint Vincent and the Grenadines',
         've': 'Venezuela',
         'vg': 'British Virgin Islands (United Kingdom)',
         'vi': 'United States Virgin Islands (United States)',
         'vn': 'Vietnam',
         'vu': 'Vanuatu',
         'wf': 'Wallis and Futuna',
         'ws': 'Samoa',
         'ye': 'Yemen',
         'yt': 'Mayotte',
         'za': 'South Africa',
         'zm': 'Zambia',
         'zw': 'Zimbabwe'}

    for x in l.keys():
        if match_case(domain, pattern=fr"\.{x}$") or match_case(domain, pattern=fr"^{x}\."):
            if return_code:
                return x
            else:
                return l[x]
    if return_code:
        return 'us'
    else:
        return l['us']


def add_url_schema(url):
    from urllib.parse import urlparse, ParseResult
    p = urlparse(url, 'https')
    netloc = p.netloc or p.path
    path = p.path if p.netloc else ''
    if not netloc.startswith('www.'):
        netloc = 'www.' + netloc

    p = ParseResult('https', netloc, path, *p[3:])
    return p.geturl()


def extract_url_netlocs(urls):
    rs = []
    for x in urls:
        rs.append(urlparse(x).netloc)
    return rs


def create_schema_from_df(df):
    schema = []
    for x in list(df):
        if match_case(x, pattern='time|_at|date|last_modified'):
            type = "DATE"
        else:
            type = "STRING"
        schema.append({"name": f"{x}", "type": type})
    return json.dumps(schema, indent=4)
