
def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')

    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # prev[num_cols].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.99]).T

    # Numerikler için describe attığımda problemli 365243 değerinin alttaki değişkenlerde bulunduğunu gördüm.

    prev.loc[:, ['DAYS_FIRST_DRAWING',
                 'DAYS_FIRST_DUE',
                 'DAYS_LAST_DUE_1ST_VERSION',
                 'DAYS_LAST_DUE',
                 'DAYS_TERMINATION', ]].replace(365243.0, np.nan, inplace=True)

    # XNA sınıfını bulunduran değişkenleri gözlemlemek istiyorum:

    for col in cat_cols:
        print()
        print("*****************************")
        print("Classes of", col, ":")
        print(prev[col].unique())

    # # Sadece XNA bulunduran değişkenleri yazdır:
    # XNA_list = []
    # for col in cat_cols:
    #     for i in range(len(prev[col].unique())):
    #         if prev[col].unique()[i] == "XNA":
    #             XNA_list.append(col)
    # print(XNA_list)
    #
    # # XNA bulunduran değişkenleri np.nan ile değiş:
    # for col in cat_cols:
    #     for i in range(len(prev[col].unique())):
    #         if prev[col].unique()[i] == "XNA":
    #             prev[col].replace("XNA", np.nan, inplace=True)

    # CAT ANALYZER

    # for i in cat_cols + cat_but_car + num_but_cat:
    #     cat_analyzer(prev, i)

    drop_list = ['FLAG_LAST_APPL_PER_CONTRACT', 'NAME_TYPE_SUITE', 'NAME_SELLER_INDUSTRY', 'NFLAG_LAST_APPL_IN_DAY',
                 'NFLAG_LAST_APPL_IN_DAY', 'WEEKDAY_APPR_PROCESS_START']

    prev.drop(drop_list, axis=1, inplace=True)

    # Rare Encoder
    rare_cols = ["NAME_PAYMENT_TYPE", "CODE_REJECT_REASON", "CHANNEL_TYPE", "NAME_GOODS_CATEGORY",
                 "PRODUCT_COMBINATION"]

    for i in rare_cols:
        rare_encoder_prev(prev, i, rare_perc=0.01)

    prev["NAME_CASH_LOAN_PURPOSE"] = np.where(~prev["NAME_CASH_LOAN_PURPOSE"].isin(["XAP", "XNA"]), "Other",
                                              prev["NAME_CASH_LOAN_PURPOSE"])

    prev.loc[(prev["NAME_PORTFOLIO"] == "Cards"), "NAME_PORTFOLIO"] = "Cars"

    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # for i in cat_cols + cat_but_car + num_but_cat:
    #     cat_analyzer(prev, i)

    # FEATURE ENGINEERING

    # kategorik değişken kırılımında target analizi

    # cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # Feature Engineering

    # FEATURE: Müşterinin istediği kredi miktarının, müşterinin aldığı kredi miktarına oranı
    prev['NEW_APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # FEATURE:

    # 10/50 ---- 1 (İSTEDİĞİNDEN DE FAZLASINI ALDI --- 1 BASILIR)
    # 60/10 ---- 0 (İSTEDİĞİNDEN AZ ALDI --- 0 BASILIR)

    prev["NEW_APP_CREDIT_RATE_PERC"] = prev["NEW_APP_CREDIT_PERC"].apply(lambda x: 1 if (x <= 1) else 0)

    # FEATURE: Kredi peşinatının yıllık ödemeye oranı
    prev['NEW_AMT_PAYMENT_ANNUITY_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_ANNUITY']

    # FEATURE: Kredi peşinatının kredi tutarına oranı
    prev['NEW_AMT_PAYMENT_CREDIT_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']

    # FEATURE: Mal fiyatının kredi tutarına oranı
    prev['NEW_GOODS_PRICE_CREDIT_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']

    # FEATURE: Müşterinin talep ettiği miktardan, talep ettiği mal fiyatının çıkarılması
    prev['NEW_APPLICATION_GOODS_SUBSTRACT'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']

    # FEATURE: Müşterinin talep ettiği miktarın, talep ettiği mal fiyatına oranı
    prev['NEW_APPLICATION_GOODS_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_GOODS_PRICE']

    # FEATURE: Peşinat oranından faiz oranının çıkarılması
    prev['NEW_RATE_PAYMENT_INTEREST_PRIMARY'] = prev['RATE_DOWN_PAYMENT'] - prev['RATE_INTEREST_PRIMARY']

    # FEATURE:
    # prev['DIFF_RATE_INTEREST_PRIVILEGED_RATE_INTEREST_PRIMARY'] = prev['RATE_INTEREST_PRIVILEGED'] - prev['RATE_INTEREST_PRIMARY']

    # FEATURE: Son vade tarihinden ilk vade tarihinin çıkarılması
    prev['NEW_LAST_AND_FIRST_SUBSTRACT'] = prev['DAYS_LAST_DUE'] - prev['DAYS_FIRST_DUE']

    # FEATURE: Önceki başvurunun sonlandırılmasından, önceki başvuru için kararın ne zaman verildiğinin çıkarılması
    prev['NEW_TERMINATION_DECISION_SUBSTRACT'] = prev['DAYS_TERMINATION'] - prev['DAYS_DECISION']

    # FEATURE: Kredi tutarının vadeye oranı
    prev["NEW_CREDIT_PAYMENT_RATIO"] = prev["AMT_CREDIT"] / prev["CNT_PAYMENT"]

    # FEATURE: Aylık ödeme miktarının yıllık ödeme miktarına oranı
    prev["NEW_CREDIT_PAYMENT_YEAR_RATIO"] = prev["NEW_CREDIT_PAYMENT_RATIO"] / prev["AMT_ANNUITY"]

    # FEATURE: Kredi ödemesinin kaç yılda biteceğinin hesabı
    prev["NEW_CREDIT_TERM_YEAR"] = prev["CNT_PAYMENT"] / 12

    # FEATURE:
    prev["NEW_CNT_PAYMENT_CAT"] = pd.cut(x=prev['CNT_PAYMENT'], bins=[0, 12, 60, 120],
                                         labels=["Short", "Middle", "Long"])

    # FEATURE: Peşinat oranı ile peşinat miktarının çarpılması
    prev['NEW_RATE_AMT_DOWN_PAYMENT'] = prev['RATE_DOWN_PAYMENT'] * prev['AMT_DOWN_PAYMENT']

    # FEATURE: Kredi tutarının faiz oranı ile çarpılması (Genele uyarlanmış biçimde)
    # prev['NEW_CREDIT_PRIMARY_RATIO'] = prev['AMT_CREDIT'] * prev['RATE_INTEREST_PRIMARY'] (nan dolu, sildik)

    # FEATURE: Kredi tutarının faiz oranı ile çarpılması (Kişiye özel uyarlanmış biçimde)
    # prev['NEW_CREDIT_PRIVILEGED_RATIO'] = prev['AMT_CREDIT'] * prev['RATE_INTEREST_PRIVILEGED'] (nan dolu, sildik)

    # FEATURE: Mal fiyatının vadeye oranı (kişinin mal için ayda ne kadar ödediğini gözlemliyoruz)
    prev['NEW_GOODS_PAYMENT_PER_MONTH'] = prev['AMT_GOODS_PRICE'] / prev['CNT_PAYMENT']

    # FEATURE: Kredinin ödeneceği yıl miktarı ile yıllık kredi ödeme miktarının çarpılması
    prev['NEW_WHOLE_CREDIT'] = prev['NEW_CREDIT_TERM_YEAR'] * prev['AMT_ANNUITY']

    # FEATURE: Toplam ödenecek kredi miktarının, kredi tutarına oranı
    prev['NEW_WHOLE_CREDIT_AMT_CREDIT_RATIO'] = prev['NEW_WHOLE_CREDIT'] / (prev['AMT_CREDIT'])

    # FEATURE: "HOUR_APPR_PROCESS_START"  değişkeninin working_hours ve off_hours olarak iki kategoriye ayrılması
    work_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    prev["NEW_HOUR_APPR_PROCESS_START"] = prev["HOUR_APPR_PROCESS_START"].replace(work_hours, 'working_hours')

    off_hours = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
    prev["NEW_HOUR_APPR_PROCESS_START"] = prev["HOUR_APPR_PROCESS_START"].replace(off_hours, 'off_hours')

    # FEATURE: X-sell approved (karşı tarafın teklifini kabul edip kredi isteği kabul edilen müşteri)
    prev['NEW_X_SELL_APPROVED'] = 0
    prev.loc[(prev['NAME_PRODUCT_TYPE'] == 'x-sell') & (
                prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_X_SELL_APPROVED'] = 1

    # FEATURE: walk-in approved (kendi istediği kredi tutarını belirtip kabul edilen müşteri)
    prev['NEW_WALK_IN_APPROVED'] = 0
    prev.loc[(prev['NAME_PRODUCT_TYPE'] == 'walk-in') & (
                prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_WALK_IN_APPROVED'] = 1

    # FEATURE: Eski müşteri olup onaylanan müşteri
    prev['NEW_REPEATER_APPROVED'] = 0
    prev.loc[(prev['NAME_CLIENT_TYPE'] == 'Repeater') & (
                prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_REPEATER_APPROVED'] = 1

    # FEATURE: Yeni müşteri olup onaylanan müşteri
    prev['NEW_NEWCUST_APPROVED'] = 0
    prev.loc[
        (prev['NAME_CLIENT_TYPE'] == 'New') & (prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_NEWCUST_APPROVED'] = 1

    # FEATURE: Kayıt yenileyip onaylanan müşteri
    prev['NEW_REFRESHED_APPROVED'] = 0
    prev.loc[(prev['NAME_CLIENT_TYPE'] == 'Refreshed') & (
                prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_REFRESHED_APPROVED'] = 1

    # # FEATURE:
    # df_prev['NEW_HIGH_APPROVED'] = 0
    # df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'high') & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_HIGH_APPROVED'] = 1
    #
    # # FEATURE:
    # df_prev['NEW_MIDDLE_APPROVED'] = 0
    # df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'middle') & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_MIDDLE_APPROVED'] = 1
    #
    # # FEATURE:
    # df_prev['NEW_LOWACTION_APPROVED'] = 0
    # df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'low_action') & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_LOWACTION_APPROVED'] = 1
    #
    # # FEATURE:
    # df_prev['NEW_LOWNORMAL_APPROVED'] = 0
    # df_prev.loc[(df_prev['NAME_YIELD_GROUP'] == 'low_normal') & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_LOWNORMAL_APPROVED'] = 1

    # credit requested / credit given ratio
    # df_prev['NEW_APP_CREDIT_RATIO'] = df_prev['AMT_APPLICATION'].div(df_prev['AMT_CREDIT']).replace(np.inf, 0) (Accuracy düşük çıkarsa buradaki yöntemi uygula)

    # prev['NEW_GOODS_PRICE_CREDIT_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']
    # # risk assessment via NEW_GOODS_PRICE_CREDIT_RATIO
    # prev.loc[prev['NEW_GOODS_PRICE_CREDIT_RATIO'] >= 1, 'NEW_CREDIT_GOODS_RISK'] = 0
    # prev.loc[prev['NEW_GOODS_PRICE_CREDIT_RATIO'] < 1, 'NEW_CREDIT_GOODS_RISK'] = 1

    # risk to approved
    # df_prev['NEW_RISK_APPROVED'] = 0
    # df_prev.loc[(df_prev['NEW_GOODS_PRICE_CREDIT_RATIO'] == 1) & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_RISK_APPROVED'] = 1

    # non risk to approved
    # df_prev['NEW_NONRISK_APPROVED'] = 0
    # df_prev.loc[(df_prev['NEW_CREDIT_GOODS_RISK'] == 0) & (df_prev['NAME_CONTRACT_STATUS'] == 'Approved'), 'NEW_NONRISK_APPROVED'] = 1

    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # high_correlation(prev, remove=['SK_ID_CURR','SK_ID_PREV'], corr_coef = "spearman", corr_value = 0.7)

    drop_list1 = ["NEW_CREDIT_PAYMENT_RATIO", "NEW_RATE_AMT_DOWN_PAYMENT", "NEW_AMT_PAYMENT_CREDIT_RATIO",
                  "NEW_GOODS_PRICE_CREDIT_RATIO", "NEW_APPLICATION_GOODS_RATIO"]

    prev.drop(drop_list1, axis=1, inplace=True)

    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max'],
        'AMT_DOWN_PAYMENT': ['min', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'mean'],
        'HOUR_APPR_PROCESS_START': ['mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'NEW_APP_CREDIT_PERC': ['min', 'mean'],
        'NEW_AMT_PAYMENT_ANNUITY_RATIO': ['min', 'max'],
        # 'NEW_AMT_PAYMENT_CREDIT_RATIO': ['min', 'max', 'mean'],
        # 'NEW_GOODS_PRICE_CREDIT_RATIO': ['max', 'min'],
        'NEW_APPLICATION_GOODS_SUBSTRACT': ['min', 'max', 'mean'],
        # 'NEW_APPLICATION_GOODS_RATIO': ['mean'],
        'NEW_RATE_PAYMENT_INTEREST_PRIMARY': ['mean'],
        'NEW_LAST_AND_FIRST_SUBSTRACT': ['min', 'max', 'mean'],
        'NEW_TERMINATION_DECISION_SUBSTRACT': ['min', 'max', 'mean'],
        # 'NEW_CREDIT_PAYMENT_RATIO': ['min', 'max', 'mean'],
        'NEW_CREDIT_PAYMENT_YEAR_RATIO': ['min', 'max'],
        'NEW_CREDIT_TERM_YEAR': ['min', 'max', 'mean'],
        # 'NEW_RATE_AMT_DOWN_PAYMENT': ['sum', 'mean'],
        'NEW_GOODS_PAYMENT_PER_MONTH': ['sum', 'max', 'min'],
        'NEW_WHOLE_CREDIT': ['sum', 'mean'],
        'NEW_WHOLE_CREDIT_AMT_CREDIT_RATIO': ['mean']}

    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    # cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(prev)

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg
