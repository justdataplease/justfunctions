## Table of Contents
- [date_functions.py](#date_functions)
  - [Function: get_date_el](#get-date-el)
  - [Function: date_to_unix](#date-to-unix)
  - [Function: unix_to_date](#unix-to-date)
  - [Function: evaluate_dates](#evaluate-dates)
  - [Function: substract_dates](#substract-dates)
  - [Function: el_to_date](#el-to-date)
  - [Function: str_to_date](#str-to-date)
  - [Function: transform_date_str_format](#transform-date-str-format)
  - [Function: date_to_str](#date-to-str)
  - [Function: to_date](#to-date)
  - [Function: add_days](#add-days)
  - [Function: today](#today)
  - [Function: start_of_span](#start-of-span)
  - [Function: add_months](#add-months)
  - [Function: dates_sequence](#dates-sequence)
  - [Function: dates_sequence_start_end_of_month](#dates-sequence-start-end-of-month)
  - [Function: dates_sequence_start_of_month](#dates-sequence-start-of-month)
  - [Function: dates_sequence_sublist](#dates-sequence-sublist)
- [nlp_functions.py](#nlp_functions)
  - [Function: website_regexp_validator](#website-regexp-validator)
  - [Function: get_digit](#get-digit)
  - [Function: website_validator](#website-validator)
  - [Function: escape_html_text](#escape-html-text)
  - [Function: remove_special_symbols](#remove-special-symbols)
  - [Function: remove_special_symbol_chars](#remove-special-symbol-chars)
  - [Function: match_case](#match-case)
  - [Function: match_all_case](#match-all-case)
  - [Function: clear_input_symbols](#clear-input-symbols)
  - [Function: clean_url_link](#clean-url-link)
  - [Function: remove_elem](#remove-elem)
  - [Function: remove_domain](#remove-domain)
  - [Function: extract_email_domain](#extract-email-domain)
  - [Function: extract_email_subdomain](#extract-email-subdomain)
  - [Function: extract_domain](#extract-domain)
  - [Function: find_country_from_domain](#find-country-from-domain)
  - [Function: add_url_schema](#add-url-schema)
  - [Function: extract_url_netlocs](#extract-url-netlocs)
  - [Function: create_schema_from_df](#create-schema-from-df)
- [general_functions.py](#general_functions)
  - [Function: list_chunks](#list-chunks)
  - [Function: request_is_redirect](#request-is-redirect)
  - [Function: get_abs_path](#get-abs-path)
  - [Function: custom_integration_retry](#custom-integration-retry)
  - [Function: print_class_attr](#print-class-attr)
  - [Function: custom_df_conditions](#custom-df-conditions)
  - [Function: fix_vname](#fix-vname)
  - [Function: define_vname_column_name](#define-vname-column-name)
  - [Function: define_vname_column_prefix](#define-vname-column-prefix)
  - [Function: get_fields](#get-fields)
  - [Function: dict_rename_key](#dict-rename-key)
  - [Function: get_schema_json](#get-schema-json)
  - [Function: series_not_null](#series-not-null)
  - [Function: check_schema](#check-schema)
  - [Function: get_unique_list_of_dicts](#get-unique-list-of-dicts)
  - [Function: no_error](#no-error)
  - [Function: discover_series_type](#discover-series-type)
  - [Function: discover_series_type_by_sample](#discover-series-type-by-sample)
  - [Function: discover_df_types](#discover-df-types)
  - [Function: discover_df_types_v2](#discover-df-types-v2)
  - [Function: write_pickle](#write-pickle)
  - [Function: open_pickle](#open-pickle)
  - [Function: normalize_df_headers](#normalize-df-headers)
  - [Function: remove_last_symbol](#remove-last-symbol)
  - [Function: remove_first_symbol](#remove-first-symbol)
  - [Function: normalize_headers](#normalize-headers)
  - [Function: list_to_string](#list-to-string)
  - [Function: flatten](#flatten)
  - [Function: list_to_str](#list-to-str)
  - [Function: stringify_list](#stringify-list)
  - [Function: chunks](#chunks)
  - [Function: round_up_to_even](#round-up-to-even)
  - [Function: accepts](#accepts)
  - [Function: parallel_jobs](#parallel-jobs)
  - [Function: run_concurrent_tasks](#run-concurrent-tasks)
  - [Function: isNaN](#isnan)
  - [Function: remove_url_tail](#remove-url-tail)
  - [Function: decode_uri](#decode-uri)
  - [Function: df_split](#df-split)
  - [Function: df_split_by_field](#df-split-by-field)
  - [Function: df_split_by_size](#df-split-by-size)
  - [Function: get_dir_files](#get-dir-files)
  - [Function: elapsed_since](#elapsed-since)
  - [Function: get_process_memory](#get-process-memory)
  - [Function: format_bytes](#format-bytes)
  - [Function: memory_profile](#memory-profile)
  - [Function: check_position](#check-position)
  - [Function: tryget](#tryget)
  - [Function: convert_size](#convert-size)
  - [Function: randomize_series](#randomize-series)
  - [Function: isinstance_df](#isinstance-df)
  - [Class: CustomIntegrationError](#customintegrationerror)
    - [Method: __init__](#--init--)
  - [Class: UniqueDict](#uniquedict)
    - [Method: __setitem__](#--setitem--)
- [csv_functions.py](#csv_functions)

---

# date_functions.py

## Function: get_date_el
```
None
```

## Function: date_to_unix
```
None
```

## Function: unix_to_date
```
None
```

## Function: evaluate_dates
```
None
```

## Function: substract_dates
```
None
```

## Function: el_to_date
```
None
```

## Function: str_to_date
```
None
```

## Function: transform_date_str_format
```
None
```

## Function: date_to_str
```
None
```

## Function: to_date
```
None
```

## Function: add_days
```
Add days to date
:param date:
:param num:
:param date_format:
:return:
```

## Function: today
```
None
```

## Function: start_of_span
```
None
```

## Function: add_months
```
Add days to date
:param date:
:param num:
:param date_format:
:return:
```

## Function: dates_sequence
```
None
```

## Function: dates_sequence_start_end_of_month
```
None
```

## Function: dates_sequence_start_of_month
```
None
```

## Function: dates_sequence_sublist
```
None
```

# nlp_functions.py

## Function: website_regexp_validator
```
None
```

## Function: get_digit
```
None
```

## Function: website_validator
```
None
```

## Function: escape_html_text
```
escape strings for display in HTML
```

## Function: remove_special_symbols
```
Remove special symbols
:param x:
```

## Function: remove_special_symbol_chars
```
None
```

## Function: match_case
```
Match pattern
:param x:
:param pattern:
:param failcase:
:param return_result:
:return:
```

## Function: match_all_case
```
Match pattern
:param x:
:param pattern:
:param failcase:
:return:
```

## Function: clear_input_symbols
```
Clear inpute symbols
:param x:
:return:
```

## Function: clean_url_link
```
Clean url link
:param url:
:param no_domain:
:return:
```

## Function: remove_elem
```
None
```

## Function: remove_domain
```
Remove domain
:param x:
:return:
```

## Function: extract_email_domain
```
None
```

## Function: extract_email_subdomain
```
None
```

## Function: extract_domain
```
Extract domain
:param url:
:return:
```

## Function: find_country_from_domain
```
None
```

## Function: add_url_schema
```
None
```

## Function: extract_url_netlocs
```
None
```

## Function: create_schema_from_df
```
None
```

# general_functions.py

## Function: list_chunks
```
Yield successive n-sized chunks from lst.
```

## Function: request_is_redirect
```
None
```

## Function: get_abs_path
```
None
```

## Function: custom_integration_retry
```
Retry decorator

@custom_integration_retry(attempts_dict={400: 3}, sleep_time_dict={400: 1}, default_attempts=5, default_sleep_time=1,
                      strategy='constant')
def test_custom_integration_retry():
    raise CustomIntegrationError({'status': 400})

test_custom_integration_retry()
```

## Function: print_class_attr
```
None
```

## Function: custom_df_conditions
```
None
```

## Function: fix_vname
```
None
```

## Function: define_vname_column_name
```
None
```

## Function: define_vname_column_prefix
```
None
```

## Function: get_fields
```
None
```

## Function: dict_rename_key
```
None
```

## Function: get_schema_json
```
get_schema_json(['services', 'google_ads', 'deps', 'schema.json'])
:param list_path:
:return:
```

## Function: series_not_null
```
None
```

## Function: check_schema
```
None
```

## Function: get_unique_list_of_dicts
```
None
```

## Function: no_error
```
None
```

## Function: discover_series_type
```
None
```

## Function: discover_series_type_by_sample
```
None
```

## Function: discover_df_types
```
None
```

## Function: discover_df_types_v2
```
None
```

## Function: write_pickle
```
None
```

## Function: open_pickle
```
None
```

## Function: normalize_df_headers
```
Normalize df colnames
:param df:
:param upper:
:return:
```

## Function: remove_last_symbol
```
None
```

## Function: remove_first_symbol
```
None
```

## Function: normalize_headers
```
None
```

## Function: list_to_string
```
None
```

## Function: flatten
```
None
```

## Function: list_to_str
```
List to string
:return:
```

## Function: stringify_list
```
List to string
:return:
```

## Function: chunks
```
None
```

## Function: round_up_to_even
```
None
```

## Function: accepts
```
None
```

## Function: parallel_jobs
```
None
```

## Function: run_concurrent_tasks
```
None
```

## Function: isNaN
```
None
```

## Function: remove_url_tail
```
Remove url tail after ?
:param x:
:return:
```

## Function: decode_uri
```
Decode utf8
:param url:
:return:
```

## Function: df_split
```
None
```

## Function: df_split_by_field
```
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
```

## Function: df_split_by_size
```
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
```

## Function: get_dir_files
```
None
```

## Function: elapsed_since
```
None
```

## Function: get_process_memory
```
None
```

## Function: format_bytes
```
None
```

## Function: memory_profile
```
None
```

## Function: check_position
```
None
```

## Function: tryget
```
None
```

## Function: convert_size
```
None
```

## Function: randomize_series
```
None
```

## Function: isinstance_df
```
None
```

## Class: CustomIntegrationError
```
Base class for other exceptions
```

### Method: __init__
```
None
```

## Class: UniqueDict
### Method: __setitem__
```
None
```

# csv_functions.py

