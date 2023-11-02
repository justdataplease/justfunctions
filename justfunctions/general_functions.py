import collections
import concurrent.futures
import html
import inspect
import json
import logging
import math
import operator
import os
import pickle
import random
import re
import sys
import time
from copy import deepcopy
from datetime import date as date_type
from datetime import datetime, timedelta
from datetime import datetime as datetime_type
from functools import wraps
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import requests
import simplejson
from dateutil import relativedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def list_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def request_is_redirect(response):
    for x in response.history:
        if x.status_code == 302:
            return True
        else:
            return False


def get_abs_path(path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


class CustomIntegrationError(Exception):
    """Base class for other exceptions"""

    def __init__(self, custom_error):
        self.status = custom_error.get('status')
        self.description = custom_error.get('error_description')
        self.description_verbose = custom_error.get('error_description_verbose')
        self.display_priority = custom_error.get('error_display_priority', 100)
        self.is_maya_integration_error = True


def custom_integration_retry(attempts_dict={}, sleep_time_dict={}, default_attempts=5, default_sleep_time=6 * 15,
                             strategy='constant'):
    """
    Retry decorator

    @custom_integration_retry(attempts_dict={400: 3}, sleep_time_dict={400: 1}, default_attempts=5, default_sleep_time=1,
                          strategy='constant')
    def test_custom_integration_retry():
        raise CustomIntegrationError({'status': 400})

    test_custom_integration_retry()
    """

    def decorator(func):
        @wraps(func)
        def newfn(*args, **kwargs):
            previous_error_status = 12345678910
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    logger.info(f"{func.__name__} / {str(exc)}")
                    error_status = exc.status
                    retry_attempts = attempts_dict.get(error_status, default_attempts)

                    # If error status changed restart attempts
                    if previous_error_status != error_status:
                        attempt = 1
                    # if attempt > retry_attempts stop
                    if attempt > retry_attempts:
                        break
                    # Select strategy
                    if strategy == 'backoff':
                        retry_sleep_time = sleep_time_dict.get(error_status, default_sleep_time) * 2 ** attempt + random.uniform(0, 1)
                    else:
                        retry_sleep_time = sleep_time_dict.get(error_status, default_sleep_time)
                    logger.info(f'Retrying : attempt {attempt}/{retry_attempts}, {retry_sleep_time}s')
                    # Sleep
                    time.sleep(retry_sleep_time)
                    attempt += 1
                    previous_error_status = error_status
            return func(*args, **kwargs)

        return newfn

    return decorator


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError("Key already exists")


def print_class_attr(attrs):
    rs = ', '.join("%s: %s" % item for item in attrs.items())
    print(rs)
    return rs


def custom_df_conditions(df, report_specs):
    ctr = report_specs['fields'].get('custom_transformations')
    if not ctr:
        return df

    ops = {
        '=': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '<': operator.lt,
        '=>': operator.le,
        '<=': operator.ge
    }
    field = ctr['field']
    condition = ctr['condition']
    value = int(ctr['value'])
    condition_func = ops[condition]

    return df[condition_func(pd.to_numeric(df[field]), value)].copy()


def fix_vname(word):
    if not word:
        return word
    return word.lower().replace(" ", "_")


def define_vname_column_name(x, x_init, dim_prefix):
    candidate = dim_prefix + x if x else dim_prefix[:-1]
    return x if '_x_id' in x_init else candidate


def define_vname_column_prefix(report_vname, report_vname_prefix):
    dim_prefix = ''
    if report_vname_prefix or report_vname[0:1] == '_':
        dim_prefix = report_vname_prefix if report_vname_prefix else report_vname[1:]
        dim_prefix += '_'
    return dim_prefix


def get_fields(columns, action=False, exclude_from=[], columns_prefix=False,
               type_map=False, return_excluded=False, keep_fields=[], ignore_fields=[]):
    rs = []
    rs_not = []
    for x in columns:
        alter_name = x.get("alias", x["name"])
        view_name = fix_vname(x.get("vname", alter_name))
        try:
            doc_text = x.get("doc", {}).get("text", "")
            doc_values = x.get("doc", {}).get("values", "")
        except Exception as exc:
            logger.error(f"get fields {exc}")
            doc_text = ""
            doc_values = ""

        if ((x.get('exclude_from', 'any') not in exclude_from) or (alter_name in keep_fields)) and (alter_name not in ignore_fields):
            if not action:
                # x.update({"alter_name": alter_name})
                j = deepcopy(x)
                j["alter_name"] = alter_name
                rs.append(j)
            elif action == 'get_names_types':
                rs.append((alter_name, type_map[x["type"]], view_name, doc_text, doc_values))
            elif action == 'get_names':
                rs.append(alter_name)
            elif action == 'get_names_view':
                rs.append(f'{alter_name} {columns_prefix}_{alter_name}')
            elif action == 'get_sql':
                rs.append(f'"{alter_name}" {type_map[x["type"]]}')
            else:
                raise Exception('get_fields : not supported...')
        else:
            if not action:
                # x.update({"alter_name": alter_name})
                j = deepcopy(x)
                j["alter_name"] = alter_name
                rs_not.append(j)
            elif action == 'get_names_types':
                rs_not.append((alter_name, type_map[x["type"]], view_name))
            elif action == 'get_names':
                rs_not.append(alter_name)
            elif action == 'get_names_view':
                rs_not.append(f'{alter_name} {columns_prefix}_{alter_name}')
            elif action == 'get_sql':
                rs_not.append(f'"{alter_name}" {type_map[x["type"]]}')
            else:
                raise Exception('get_fields : not supported...')

    if return_excluded:
        return rs_not
    return rs


def dict_rename_key(d, old_key, new_key):
    return {new_key if k == old_key else k: v for k, v in d.items()}


def get_schema_json(list_path):
    """
    get_schema_json(['services', 'google_ads', 'deps', 'schema.json'])
    :param list_path:
    :return:
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(project_root)
    dir = os.path.join(base_dir, *list_path)
    with open(dir) as f:
        return json.load(f)


def series_not_null(dt):
    dt_c = dt[dt.notnull() & dt.replace([np.inf, -np.inf], np.nan).notna() & (dt.astype(str).str.strip() != '')].copy()
    return dt_c.reset_index(drop=True)


def check_schema(df, report_specs, fill_missing_type=False, truncate_missing=False):
    common_fields = report_specs['report_schema']['columns']
    common_names = [f['name'] for f in common_fields]
    for x in df.columns:
        if x not in common_names:
            if fill_missing_type:
                logger.info(f'{x} added to schema {common_names}...')
                common_fields.append({'name': x, 'type': fill_missing_type})
            elif truncate_missing:
                logger.info(f'{x} dropped from schema...')
                df.drop(x, axis=1, inplace=True)
            else:
                raise Exception(f'{x} does not exist is schema {common_names}...')
    return common_fields


def get_unique_list_of_dicts(L):
    return [dict(s) for s in set(frozenset(d.items()) for d in L)]


def no_error(func, **kwargs):
    try:
        func(**kwargs)
        return True
    except:
        return False


def discover_series_type(s):
    # df = pd.DataFrame({'a': [100, 3, 4], 'b': [20.1, 2.3, 45.3], 'c': [None, 'a', 1.00],
    #                    'd': ['27/05/2001', '1999-01-01', '25/09/1998'], 'e': [True, False, True]})
    # s = df['a']

    s_clean = series_not_null(s)
    t = s_clean.dtype
    # Get type from dtype
    if issubclass(t.type, np.floating) or issubclass(t.type, np.integer) or issubclass(t.type, np.int64):
        sqltype = 'NUMERIC'
    elif no_error(pd.to_numeric, arg=s_clean, errors='raise'):
        sqltype = 'NUMERIC'
    elif issubclass(t.type, np.datetime64) or no_error(pd.to_datetime, arg=s_clean, errors='raise', exact=False):
        try:
            check_condition = pd.to_datetime(s_clean, errors='coerce', exact=False).dt.strftime(
                '%H:%M:%S') == '00:00:00'
            sqltype = 'DATE' if (check_condition[0] and check_condition.all()) else 'TIMESTAMP'
        except Exception as exc:
            sqltype = 'TIMESTAMP'
    elif issubclass(t.type, np.bool_):
        sqltype = 'BOOLEAN'
    elif issubclass(t.type, list):
        sqltype = 'LIST_AS_STR'
    else:
        sqltype = 'STRING'

    return sqltype


def discover_series_type_by_sample(s):
    s_clean = series_not_null(s)
    counter = 0
    sqltype_solutions = []
    while counter <= 3 and counter <= len(s_clean) - 1:
        sampl = s_clean[counter]
        try:
            float(sampl)
            sqltype_solutions.append('NUMERIC')
        except Exception as exc:
            if isinstance(sampl, str):
                sqltype_solutions.append('STRING')
            elif isinstance(sampl, date_type):
                sqltype_solutions.append('DATE')
            elif isinstance(sampl, datetime_type):
                sqltype_solutions.append('TIMESTAMP')
            elif isinstance(sampl, bool):
                sqltype_solutions.append('BOOLEAN')
            elif isinstance(sampl, list):
                sqltype_solutions.append('ARRAY_AS_STR')
            elif isinstance(sampl, dict):
                sqltype_solutions.append('DICT_AS_STR')
            else:
                logger.error(type(sampl))
        finally:
            counter += 1

    # Decide sample type
    logger.info(sqltype_solutions)
    if len(set(sqltype_solutions)) == 1:
        return sqltype_solutions[0]
    else:
        return 'STRING'

    return sqltype


def discover_df_types(df, by_sample=False):
    column_types = {}
    for k in df.columns:
        if by_sample:
            column_types[k] = {'type': discover_series_type_by_sample(df[k])}
        else:
            column_types[k] = {'type': discover_series_type(df[k])}
    return column_types


def discover_df_types_v2(df, by_sample=False):
    column_types = []
    for k in df.columns:
        if by_sample:
            column_types.append({'name': k, 'type': discover_series_type_by_sample(df[k])})
        else:
            column_types.append({'name': k, 'type': discover_series_type(df[k])})
    return column_types


def write_pickle(data, filename):
    pickle.dump(data, filename)


def open_pickle(filename):
    data = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data


def normalize_df_headers(df):
    """
    Normalize df colnames
    :param df:
    :param upper:
    :return:
    """

    colnames = list(df)

    df.columns = normalize_headers(k=colnames)
    logger.info(f'headers normalized : {df.columns}')
    return


def remove_last_symbol(iterable, symbol='_'):
    if iterable[-1] == symbol:
        return iterable[:len(iterable) - 1]
    else:
        return iterable


def remove_first_symbol(iterable, symbol='_'):
    if iterable[0] == symbol:
        return iterable[1:]
    else:
        return iterable


def normalize_headers(k):
    def normalize_item(x, bad_symbols, snake_case):
        k0 = snake_case.sub(r'_\1', x).lower()
        # Remove special symbols
        k1 = bad_symbols.sub('_', k0)
        # Strip & Replace Spaces
        k2 = re.sub('\s+', '_', k1.strip())
        # Replace multiple underscores
        k3 = re.sub('_+', '_', k2)
        # Replace last letter if symbol
        k3 = remove_last_symbol(k3, "_")
        k3 = remove_first_symbol(k3, "_")

        return k3

    """
    Fixes headers format
    :param list:
    """
    bad_symbols = re.compile(r'[^\w\s]+', flags=re.IGNORECASE)
    snake_case = re.compile(r'((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')

    if isinstance(k, list):
        rs = []
        for x in k:
            rs.append(normalize_item(x, bad_symbols, snake_case))
        return rs
    elif isinstance(k, str):
        return normalize_item(k, bad_symbols, snake_case)
    else:
        logger.warning(f'{k} not supported...')


def list_to_string(x):
    try:
        if not x:
            return ''
        else:
            return ','.join(map(str, x))
    except Exception:
        pass


def flatten(d, parent_key='', sep='_', list_max_elements=0):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            counter = 0
            for index, elem in enumerate(v):
                if isinstance(elem, collections.MutableMapping):
                    items.extend(flatten(elem, f"{new_key}_{counter}", sep=sep).items())
                    counter += 1
                    if counter > list_max_elements:
                        break
                else:
                    items.append((new_key, elem))
        else:
            items.append((new_key, v))
    return dict(items)


def list_to_str(lst):
    """
    List to string
    :return:
    """
    return ', '.join(map(lambda x: "'" + str(x) + "'", lst))


def stringify_list(lst):
    """
    List to string
    :return:
    """
    if not lst:
        return None
    return simplejson.dumps(lst, ignore_nan=True, default=str)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def round_up_to_even(f):
    if f == 1:
        return 1
    return math.ceil(f / 2.) * 2


def accepts(*types):
    def check_accepts(f):
        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                    "arg %r does not match %s" % (a, t)
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts


def parallel_jobs(items, func, var1, connections=5):
    with concurrent.futures.ProcessPoolExecutor(max_workers=connections) as executor:
        task_queue = [executor.submit(func, item, var1) for item in items]
        for task in concurrent.futures.as_completed(task_queue):
            try:
                task.result()
            except:
                yield task.result()


def run_concurrent_tasks(func, items, connections=5, *args):
    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as executor:
        task_queue = [executor.submit(func, item, *args) for item in items]
        for task in concurrent.futures.as_completed(task_queue):
            try:
                logger.info(task.result())
            except Exception as exc:
                logger.info(exc)


def isNaN(num):
    return num != num


def remove_url_tail(x):
    """
    Remove url tail after ?
    :param x:
    :return:
    """
    return re.sub(r'\?.+|\#.+', '', x)


def decode_uri(url):
    """
    Decode utf8
    :param url:
    :return:
    """
    if isinstance(url, str):
        try:
            url = unquote(url)
        except Exception as exc:
            logger.error('decode_uri : ' + str(exc))
            url = ''
        return url
    else:
        return ''


def df_split(df, n=1000):
    for i in range(0, df.shape[0], n):
        yield df[i:i + n]


def df_split_by_field(df, field='date', backfill=False, ignore=False):
    """
        import pandas as pd

        df = pd.DataFrame(
            {
                "Company": [
                    "Samsung", "Samsung", "Samsung", "Samsung", "Samsung", "LG", "LG", "LG", "LG", "LG", "Sony", "Sony", "Sony",
                    "Sony", "Sony",
                ],
                "date": [
                    "10/9/2015", "10/9/2015", "10/9/2017", "10/10/2017", "10/10/2017", "10/10/2018", "10/9/2018", "10/9/2018",
                    "10/9/2018", "10/10/2016", "10/10/2016", "10/10/2016", "10/10/2019", "10/10/2019", "10/10/2019",
                ],
                "Country": [
                    "India", "India", "USA", "France", "India", "India", "Germany", "USA", "Brazil", "Brazil", "India",
                    "Germany", "India", "India", "Brazil",
                ],
                "Sells": [15, 81, 29, 33, 21, 42, 67, 35, 2, 34, 21, 50, 10, 26, 53],
            }
        )


        for x in df_split_by_field(df):
            for y in df_split_by_size(x):
                print(y)
    :param df:
    :param filed:
    :return:
    """
    if ignore:
        yield {'field': '', 'data': df}
    else:
        if backfill:
            df.sort_values(field, ascending=False, inplace=True)
        else:
            df.sort_values(field, ascending=True, inplace=True)

        for x, y in df.groupby(field, as_index=False, sort=False):
            yield {'field': x, 'data': pd.DataFrame(y)}


def df_split_by_size(df, mb=1000):
    """
    Generator function that splits dataframe into chunks based on megabytes
    Example:
    df = pd.DataFrame(np.random.randint(0, 100, size=(100000000, 4)), columns=list('ABCD'))
    sys.getsizeof(df)/1000000

    rs = []
    for rdf in df_split_by_size(df, mb=10):
        rs.append(rdf)

    adfs = pd.concat(rs)
    df.equals(adfs)

    :param df: pandas dataframe
    :param mb: megabytes
    :return: pandas dataframe chunk
    """
    bmb = 1000000
    size = sys.getsizeof(df)
    # size = df.memory_usage(deep=True).sum()
    if size / bmb >= mb:
        logger.info(f'size exceeded {mb}mb')
        ndfs = np.array_split(df, 2)
        for ndf in ndfs:
            yield from df_split_by_size(ndf, mb=mb)
    else:
        yield df


def get_dir_files(path, pattern):
    files = []
    for file in os.listdir(path):
        if file.endswith(pattern):
            files.append(file)
    return files


def elapsed_since(start):
    # return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    elapsed = time.time() - start
    if elapsed < 1:
        return str(round(elapsed * 1000, 2)) + "ms"
    if elapsed < 60:
        return str(round(elapsed, 2)) + "s"
    if elapsed < 3600:
        return str(round(elapsed / 60, 2)) + "min"
    else:
        return str(round(elapsed / 3600, 2)) + "hrs"


def get_process_memory():
    import psutil
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms


def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes) + "B"
    elif abs(bytes) < 1e6:
        return str(round(bytes / 1e3, 2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


def memory_profile(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        rss_before, vms_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        rss_after, vms_after = get_process_memory()
        print("Profiling: {:>20}  RSS: {:>8} | VMS: {:>8} | time: {:>8}"
              .format("<" + func.__name__ + ">",
                      format_bytes(rss_after - rss_before),
                      format_bytes(vms_after - vms_before),
                      elapsed_time))
        return result

    if inspect.isfunction(func):
        return wrapper
    elif inspect.ismethod(func):
        return wrapper(*args, **kwargs)


def check_position(a, pos, ret=None):
    try:
        r = a[pos]
    except IndexError:
        r = ret
    except KeyError:
        r = ret
    return r


def tryget(x, name, failcase=None):
    if x is None:
        return failcase
    try:
        if x[name]:
            return x[name]
        else:
            return failcase
    except (KeyError, AttributeError):
        return failcase


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def randomize_series(s, ttype, params):
    random_factor = params.get('random_factor')

    if random_factor == 1:
        return s

    np.random.seed(1234)
    if ttype == "numeric":
        if random_factor:
            multiplier = random_factor
        else:
            multiplier = np.random.uniform(low=0.2, high=1.5)

        return s.apply(lambda x: int(x * multiplier)).copy()
    elif ttype == "string":
        def complete_rule(rule):
            return "|".join([f"\w*{x}\w*" for x in rule.split("|")])

        rules = {"cryptotrade": "polfi|pollfi|polefi|plfi|pllfi|poll fi|pol fi|poolf|pooo|polif",
                 "trading": "research",
                 "investor ": "customer",
                 "coin": "response",
                 "cardano": "focus group|focus|segment",
                 "coinbase wallet": "google poll ",
                 "coinbase": "google|gogle|voting",
                 "marketplace": "branded",
                 "blockchain plan": " plan ",
                 "blockchain": " poll ",
                 "market": "market|markt|form",
                 "crypto": "surv|srve|quest|qust|chart|visual|dashb|descr|brand",
                 "platform": "satisfa|nps|feedback",
                 "profit": "score",
                 "loss": "error|bias",
                 "decentralized": "online|open",
                 "quantitative": "quant|qualit",
                 "money": "monney|money",
                 "cryptocom": "monkey",
                 "kraken": "agoda",
                 "etherium": "typeform",
                 "nfts": "example|exper|sampl|smpl",
                 "reinvest": "calcul|clcu|scale|conduct",
                 "bitcoin": "respond|docs",
                 "increase": "promote",
                 "earnings": "proba|distri|dstri",
                 "news": "aware",
                 "impact": "correl|corel|statistic|confid",
                 "bull": "statistic|pump and dump",
                 "rank": "rank|interval|intrval",
                 "cryptowallet": "answer|answ",
                 "ether": "explorat|resource|explan",
                 "analysis": "analy|anly",
                 "ethereum": "panel|blog",
                 "probability": "rating",
                 "metaverse": "design",
                 "helium": "likert",
                 "invest": "consume"
                 }

        for i in list(rules):
            regex_pat = re.compile(r"{rules}".format(rules=complete_rule(rules[i])), flags=re.IGNORECASE)
            s = s.str.replace(regex_pat, i, regex=True)

        return s


def isinstance_df(input_df):
    return input_df if isinstance(input_df, pd.DataFrame) else pd.DataFrame()
