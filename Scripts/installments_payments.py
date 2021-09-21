
def installments_payments(num_rows=None, nan_as_category=True):
    import pandas as pd
    import numpy as np
    ins = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv',nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category)
    ins['NEW_PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']  # Kredi ödeme yüzdesi
    ins['NEW_PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']  # Toplam kalan borç
    # Days past due and days before due (no negative values)
    ins['NEW_DPE'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT'] # Ödeme tarihinden ne kadar önce ya da sonra ödedi
    ins['NEW_DPE'] = ins['NEW_DPE'].map(lambda x: 1 if x < 0 else 0) # Ödeme tarihini geçti mi geçmedi mi (1: geç ödedi, 0: erken ödedi)
    aggregations = {'NUM_INSTALMENT_VERSION': ['nunique'],
                    'NUM_INSTALMENT_NUMBER': ['max', 'mean', 'sum', 'median', 'std'],
                    'DAYS_INSTALMENT': ['max', 'mean', 'sum', 'std'],
                    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
                    'AMT_INSTALMENT': ['min','max', 'mean', 'sum', 'std'],
                    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
                    'NEW_DPE': ['max', 'mean', 'sum', 'median', 'std'],
                    'NEW_PAYMENT_PERC': ['max', 'mean', 'sum', 'std', 'median'],
                    'NEW_PAYMENT_DIFF': ['max', 'mean', 'sum', 'std', 'median']}
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    return ins_agg
