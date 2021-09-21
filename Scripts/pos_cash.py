
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv', nrows=num_rows)
    pos["NAME_CONTRACT_STATUS"] = np.where(~(pos["NAME_CONTRACT_STATUS"].isin(['Active', 'Completed', 'Signed'])),"Rare", pos["NAME_CONTRACT_STATUS"])
    pos, cat_cols = one_hot_encoder(pos, nan_as_category)

    # Features
    aggregations = {'MONTHS_BALANCE': ['min', 'max'],
                    'CNT_INSTALMENT': ['min', 'max', 'std', 'median'],
                    'CNT_INSTALMENT_FUTURE': ['min', 'max', 'std', 'median'],
                    'SK_DPD': ['max', 'mean'],
                    'SK_DPD_DEF': ['max', 'mean'],
                    'NAME_CONTRACT_STATUS_Active': ['mean','sum'],
                    'NAME_CONTRACT_STATUS_Completed': ['mean','sum'],
                    'NAME_CONTRACT_STATUS_Signed': ['mean','sum'],
                    'NAME_CONTRACT_STATUS_Rare': ['mean','sum']}

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # 1:kredi zamaninda kapanmamis 0:kredi zamaninda kapanmis
    # POS_CNT_INSTALMENT_FUTURE: ÖNCEKİ KREDİNİN ÖDENMESİ İÇİN KALAN TAKSİTLER
    # POS_NAME_CONTRACT_STATUS: AY BOYUNCA SÖZLEŞME DURUMU
    # ÖNCEKİ KREDİDE KALAN TAKSİT SIFIRA EŞİTSE VE TAMAMLANMIŞ KREDİSİ SIFIRA EŞİTSE KREDİ ZAMANINDA TAMAMLANMAMIŞ
    pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'] = (pos_agg['POS_CNT_INSTALMENT_FUTURE_MIN'] == 0) & (pos_agg['POS_NAME_CONTRACT_STATUS_Completed_SUM'] == 0)
    pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME'] = [1 if i == True else 0 for i in pos_agg['POS_NEW_IS_CREDIT_NOT_COMPLETED_ON_TIME']]

    # # Count pos cash accounts
    pos_agg['POS_NEW_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
