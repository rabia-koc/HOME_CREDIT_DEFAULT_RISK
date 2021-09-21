
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv', nrows= num_rows)
    cc["NAME_CONTRACT_STATUS"] = np.where(~(cc["NAME_CONTRACT_STATUS"].isin(['Active', 'Completed', 'Signed'])), "Rare",cc["NAME_CONTRACT_STATUS"])
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

    grp = cc.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index=str, columns={'SK_ID_PREV': 'NUMBER_OF_LOANS_PER_CUSTOMER'})
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')
    grp = cc.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index=str,columns={'CNT_INSTALMENT_MATURE_CUM': 'NUMBER_OF_INSTALMENTS'})
    grp1 = grp.groupby(by=['SK_ID_CURR'])['NUMBER_OF_INSTALMENTS'].sum().reset_index().rename(index=str, columns={'NUMBER_OF_INSTALMENTS': 'TOTAL_INSTALMENTS_OF_ALL_LOANS'})
    cc = cc.merge(grp1, on=['SK_ID_CURR'], how='left')
    cc['INSTALLMENTS_PER_LOAN'] = (cc['TOTAL_INSTALMENTS_OF_ALL_LOANS'] / cc['NUMBER_OF_LOANS_PER_CUSTOMER']).astype('uint32')

    def day_past_due(dpd):
        # Önceki ayda vadeyi geçen gün sayısının sıfıra eşit olmaması durumlarını toplar
        dpd_list = dpd.tolist()
        days = 0
        for i in dpd_list:
            if i != 0:
                days = days + 1
        # Toplamda vadenin kaç kere geciktirildiğini ifade eder
        return days
#4
    grp = cc.groupby(by=['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: day_past_due(x.SK_DPD)).reset_index().rename(index=str, columns={0: 'NUMBER_OF_DPD'})
    grp1 = grp.groupby(by=['SK_ID_CURR'])['NUMBER_OF_DPD'].mean().reset_index().rename(index=str, columns={'NUMBER_OF_DPD': 'DPD_COUNT'})

    cc = cc.merge(grp1, on=['SK_ID_CURR'], how='left')

    def min_rate(min_pay, total_pay):
        # minimum ödenmesi gereken miktardan daha az ödenmiş olan ayların yüzdeliğini hesaplar
        minimum = min_pay.tolist()
        total = total_pay.tolist()
        transactions = 0
        for i in range(len(minimum)):
            if total[i] < minimum[i]:
                transactions = transactions + 1

        return (transactions * 100) / len(minimum)

    grp = cc.groupby(by=['SK_ID_CURR']).apply(lambda x: min_rate(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index=str, columns={0: 'PERCENTAGE_MIN_MISSED_PAYMENTS'})
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')
    cc.head()

    #############################

    grp = cc.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index=str, columns={
        'AMT_DRAWINGS_ATM_CURRENT': 'DRAWINGS_ATM'})  # Önceki kredide ATM'den çekilen tutar
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    grp = cc.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={
        'AMT_DRAWINGS_CURRENT': 'DRAWINGS_TOTAL'})  # Önceki kredide çekilen tutar
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    cc['CASH_CARD_RATIO1'] = (cc['DRAWINGS_ATM'] / cc['DRAWINGS_TOTAL']) * 100  # ATM den cektigi nakit / toplam cektigi
    del cc['DRAWINGS_ATM']
    del cc['DRAWINGS_TOTAL']

    grp = cc.groupby(by=['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index=str, columns={'CASH_CARD_RATIO1': 'CASH_CARD_RATIO'})
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    grp = cc.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'TOTAL_DRAWINGS'})  # Önceki kredinin olduğu ay boyunca çekilen miktar (toplamı)
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    grp = cc.groupby(by=['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={'CNT_DRAWINGS_CURRENT': 'NUMBER_OF_DRAWINGS'})  # Önceki kredide bu aydaki çekimlerin sayısı (toplamı)
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    cc['DRAWINGS_RATIO1'] = (cc['TOTAL_DRAWINGS'] / cc['NUMBER_OF_DRAWINGS']) * 100  # yüzdelik olarak ifade edilmiyor, genişletme yapıldı
    del cc['TOTAL_DRAWINGS']
    del cc['NUMBER_OF_DRAWINGS']

    grp = cc.groupby(by=['SK_ID_CURR'])['DRAWINGS_RATIO1'].mean().reset_index().rename(index=str, columns={'DRAWINGS_RATIO1': 'DRAWINGS_RATIO'})
    cc = cc.merge(grp, on=['SK_ID_CURR'], how='left')

    del cc['DRAWINGS_RATIO1']

    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ["sum", "mean"],
        'AMT_BALANCE': ["sum", "mean", "min", "max", 'std'],
        'AMT_CREDIT_LIMIT_ACTUAL': ["sum", "mean"],

        'AMT_DRAWINGS_ATM_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_DRAWINGS_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_DRAWINGS_OTHER_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_DRAWINGS_POS_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_INST_MIN_REGULARITY': ["sum", "mean", "min", "max", 'std'],
        'AMT_PAYMENT_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_PAYMENT_TOTAL_CURRENT': ["sum", "mean", "min", "max", 'std'],
        'AMT_RECEIVABLE_PRINCIPAL': ["sum", "mean", "min", "max", 'std'],
        'AMT_RECIVABLE': ["sum", "mean", "min", "max", 'std'],
        'AMT_TOTAL_RECEIVABLE': ["sum", "mean", "min", "max", 'std'],

        'CNT_DRAWINGS_ATM_CURRENT': ["sum", "mean"],
        'CNT_DRAWINGS_CURRENT': ["sum", "mean", "max"],
        'CNT_DRAWINGS_OTHER_CURRENT': ["mean", "max"],
        'CNT_DRAWINGS_POS_CURRENT': ["sum", "mean", "max"],
        'CNT_INSTALMENT_MATURE_CUM': ["sum", "mean", "max", "min", 'std'],
        'SK_DPD': ["sum", "mean", "max"],
        'SK_DPD_DEF': ["sum", "mean", "max"],

        'NAME_CONTRACT_STATUS_Active': ["sum", "mean", "min", "max", 'std'],
        'INSTALLMENTS_PER_LOAN': ["sum", "mean", "min", "max", 'std'],

        'DPD_COUNT': ["mean"],
        'PERCENTAGE_MIN_MISSED_PAYMENTS': ["mean"],
        'CASH_CARD_RATIO': ["mean"],
        'DRAWINGS_RATIO': ["mean"]})

    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    return cc_agg
