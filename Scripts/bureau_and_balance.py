
def bureau_and_balance(num_rows=None, nan_as_category=True):
    br = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
    bb = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(br)

    # XNA bulunduran yok.
    XNA_list = []
    for col in cat_cols:
        for i in range(len(br[col].unique())):
            if br[col].unique()[i] == "XNA":
                XNA_list.append(col)

    print(XNA_list)

    # br[num_cols].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.99]).T

    # for col in cat_cols:
    #     cat_summary(br, col)

    # CORRELATION
    # high_correlation(br)

    # RARE ENCODER
    br.loc[br["CREDIT_ACTIVE"] == "Sold", "CREDIT_ACTIVE"] = "Closed"
    br.loc[br["CREDIT_ACTIVE"] == "Bad debt", "CREDIT_ACTIVE"] = "Closed"

    br.drop("CREDIT_CURRENCY", axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(br)

    br = rare_encoder(br, 0.01)

    # rare_analyser_new(br, cat_cols)

    # FEATURE ENGINEERING

    # FEATURE: Kredi başvurusu bilgilerinin ne kadar sürede geldiği (Örn 188 gün içinde geldi)
    br['NEW_DAYS_CREDIT_UPDATE_SUBSTRACT'] = br['DAYS_CREDIT'] - br['DAYS_CREDIT_UPDATE']

    # FEATURE: Kişi kaç gün erken veya kaç gün geç ödemiş (kredinin kapanması gereken zaman - kredinin kapandığı zaman)
    br["NEW_CREDIT_PAID_SUBSTRACT"] = br["DAYS_CREDIT_ENDDATE"] - br["DAYS_ENDDATE_FACT"]

    # FEATURE: Kredinin kapandığı zaman <= Kredinin bitmesine ne kadar kaldığı (yarın hocaya sorulacak)
    # br['NEW_EARLY_PAID'] = (br['DAYS_ENDDATE_FACT'] <= br['DAYS_CREDIT_ENDDATE']).astype('float')
    # Geçliği ve erkenliği ayrı ayrı ifade eden kod satırları:
    br["NEW_LATE"] = br["NEW_CREDIT_PAID_SUBSTRACT"].apply(lambda x: 1 if x < 0 else 0)  # Gecikme olup olmaması
    br["NEW_EARLY"] = br["NEW_CREDIT_PAID_SUBSTRACT"].apply(lambda x: 1 if x > 0 else 0)  # Erken ödenip ödenmeme

    # FEATURE: Geciken kredi tutarının toplam kredi tutarına oranı
    br['NEW_OVERDUE_CREDIT_SUM_PERC'] = br['AMT_CREDIT_SUM_OVERDUE'] / br['AMT_CREDIT_SUM']

    # FEATURE: Mevcut borcun tüm kredi tutarına oranı
    br['NEW_DEBT_SUM_TO_CREDIT_SUM_RATIO'] = br['AMT_CREDIT_SUM_DEBT'] / br[
        'AMT_CREDIT_SUM']  # bunu bahara söyle amt_credit_sum +1 yapan var.

    # FEATURE: Mevcut kredi mikarı - Mevcut borç = Ödenen kredi miktarı
    br["NEW_PAID_CREDIT"] = br["AMT_CREDIT_SUM"] - br["AMT_CREDIT_SUM_DEBT"]

    # FEATURE: Mevcut kredi miktarı / mevcut borç
    br['NEW_DEBT_CREDIT_RATIO'] = br['AMT_CREDIT_SUM'] / br['AMT_CREDIT_SUM_DEBT']

    # FEATURE: Ödenen kredi miktarının yüzdesi
    br["NEW_PAID_CREDIT_PERC"] = (br["NEW_PAID_CREDIT"] / br["AMT_CREDIT_SUM"]) * 100

    # FEATURE: Gecikmiş tutarın uzatma miktarına oranı
    br['NEW_CREDIT_OVERDUE_PROLONG_SUM'] = [x / y if y != 0 else 0 for x, y in
                                            br[['AMT_CREDIT_SUM_OVERDUE', 'CNT_CREDIT_PROLONG']].values]

    # FEATURE: Maksimum gecikme tutarının uzatma miktarına oranı
    br['NEW_CREDIT_OVERDUE_PROLONG_MAX'] = [x / y if y != 0 else 0 for x, y in
                                            br[['AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG']].values]

    # FEATURE: Geciktirmeden ödediği tutar
    br['NEW_CREDIT_OVERDUE_SUBSTRACT'] = br['AMT_CREDIT_SUM'] - br['AMT_CREDIT_SUM_OVERDUE']

    # FEATURE: Kredi tutarının gecikme tutarına oranı
    br['NEW_CREDIT_OVERDUE_RATIO'] = br['AMT_CREDIT_SUM'] / br['AMT_CREDIT_SUM_OVERDUE']

    # FEATURE: Kredi Bürosunda bildirilen kredi kartının mevcut limiti - Kredi Bürosu kredisinde mevcut borç
    # br['NEW_AMT_CREDIT_DEBT_SUBSTRACT'] = br['AMT_CREDIT_SUM_LIMIT'] - br['AMT_CREDIT_SUM_DEBT']

    # FEATURE:
    # br['NEW_AMT_CREDIT_DEBT_RATIO'] = br['AMT_CREDIT_SUM_DEBT'] / br['AMT_CREDIT_SUM_LIMIT']

    # FEATURE:
    br["NEW_HAS_CREDIT_CARD"] = br["AMT_CREDIT_SUM_LIMIT"].apply(lambda x: 1 if x > 0 else 0)

    # Aylık Ödeme Oranı
    br['NEW_AMT_ANNUITY_RATİO'] = br['AMT_ANNUITY'] / br['AMT_CREDIT_SUM']  # bunu sonradan ekledim kesinlikle bak .

    # FEATURE:
    br.loc[br['CREDIT_ACTIVE'] == "Closed", "NEW_IS_ACTIVE_CREDIT"] = 0
    br.loc[br['CREDIT_ACTIVE'] == "Active", "NEW_IS_ACTIVE_CREDIT"] = 1
    br["NEW_IS_ACTIVE_CREDIT"] = br["NEW_IS_ACTIVE_CREDIT"].astype("int")

    # FEATURE: Müşterinin şimdiye kadar yaptığı toplam kredi başvuru sayısı (olumlu olumsuz başvurular dahil)
    group = br[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(
        index=str, columns={'DAYS_CREDIT': 'NEW_BUREAU_LOAN_COUNT'})
    br = br.merge(group, on=['SK_ID_CURR'], how='left')  # ana tablo ile birleştirme.

    # FEATURE: Müşterinin şimdiye kadar yaptığı kredi başvurularının türünün sayısı
    group = br[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(
        index=str, columns={'CREDIT_TYPE': 'NEW_BUREAU_LOAN_TYPES'})
    br = br.merge(group, on=['SK_ID_CURR'], how='left')

    # FEATURE: Müşterinin kredi türü başına düşen başvuru sayısı. "Müşteri farklı türlerde kredi almış mı, yoksa tek bir çeşit kredi mi kullanmış", bunu gözlemliyoruz.
    br['NEW_AVERAGE_LOAN_TYPE'] = br['NEW_BUREAU_LOAN_COUNT'] / br['NEW_BUREAU_LOAN_TYPES']

    # FEATURE: Müşteri başına aktif kredilerin ortalama sayısı
    # df['CREDIT_ACTIVE_BINARY'] = df['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Active' else 0)

    br.loc[br['CREDIT_ACTIVE'] == "Closed", 'CREDIT_ACTIVE_BINARY'] = 0
    br.loc[br['CREDIT_ACTIVE'] != "Closed", 'CREDIT_ACTIVE_BINARY'] = 1
    br['CREDIT_ACTIVE_BINARY'] = br['CREDIT_ACTIVE_BINARY'].astype('int32')

    # Kapanmamış kredi borçları 1'e daha yakın ise bu iyi değildir.
    group = br.groupby(by=['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ACTIVE_BINARY': 'NEW_ACTIVE_LOANS_PERCENTAGE'})
    br = br.merge(group, on=['SK_ID_CURR'], how='left')
    del br['CREDIT_ACTIVE_BINARY']
    gc.collect()

    # FEATURE: HER MÜŞTERİ İÇİN BAŞARILI GEÇMİŞ BAŞVURULAR ARASINDAKİ ORTALAMA GÜN SAYI
    # Müşteri geçmişte ne sıklıkla kredi aldı? Düzenli zaman aralıklarında mı dağıtıldı - iyi bir finansal planlamanın işareti mi yoksa krediler daha küçük bir zaman çerçevesi etrafında mı yoğunlaştı - potansiyel finansal sıkıntıyı mı gösteriyor?

    # Her Müşteriye göre gruplandırıldı ve DAYS_CREDIT değerleri artan düzende sıralandı.
    # Kredi DAYS_CREDIT'i SK_ID_CURR bazında sıralayarak NEW_DAYS_DIFF değişkeni üretmek kredi alma frekansı bilgisi verebilir.
    grp = br[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(drop=True)
    # rename(index = str, columns = {'DAYS_CREDIT': 'DAYS_CREDIT_DIFF'})
    print("Grouping and Sorting done")

    # Calculate Difference between the number of Days
    grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT'] * -1
    grp1['NEW_DAYS_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])[
        'DAYS_CREDIT1'].diff()  # aldığı farklı krediler arasında kaçar gün olduğu hesaplandı
    grp1['NEW_DAYS_DIFF'] = grp1['NEW_DAYS_DIFF'].fillna(0).astype(
        'uint32')  # ilk değişkende nan geleceği için 0 ile doldurdum. diff fonksiyonunda 2. değerden 1. değer çıkarılıyor. bu sebeple ilk değerde nan geliyor.
    del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT'], grp1['SK_ID_CURR']
    gc.collect()

    print("Difference days calculated")
    br = br.merge(grp1, on=['SK_ID_BUREAU'], how='left')
    print("Difference in Dates between Previous CB applications is CALCULATED")

    # Feature :Ödemesi devam eden kredi sayılarının ortalaması

    br.loc[br['DAYS_CREDIT_ENDDATE'] < 0, "CREDIT_ENDDATE_BINARY"] = 0  # ödemesi bitmiş (Closed) krediler
    br.loc[br['DAYS_CREDIT_ENDDATE'] >= 0, "CREDIT_ENDDATE_BINARY"] = 1  # ödemesi devam eden (Active) krediler
    grp = br.groupby(by=['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={
        'CREDIT_ENDDATE_BINARY': 'NEW_CREDIT_ENDDATE_PERCENTAGE'})

    br = br.merge(grp, on=['SK_ID_CURR'], how='left')
    del br['CREDIT_ENDDATE_BINARY']
    gc.collect()

    # FEATURE 7
    # AVERAGE NUMBER OF DAYS IN WHICH CREDIT EXPIRES IN FUTURE -INDICATION OF CUSTOMER DELINQUENCY IN FUTURE??
    # Repeating Feature 6 to Calculate all transactions with ENDATE as POSITIVE VALUES

    br['CREDIT_ENDDATE_BINARY'] = br['DAYS_CREDIT_ENDDATE']
    # Dummy column to calculate 1 or 0 values. 1 for Positive CREDIT_ENDDATE and 0 for Negat
    br.loc[(br["DAYS_CREDIT_ENDDATE"] <= 0), "CREDIT_ENDDATE_BINARY"] = 0  # ödemesi bitmiş (Closed) krediler
    br.loc[(br["DAYS_CREDIT_ENDDATE"] > 0), "CREDIT_ENDDATE_BINARY"] = 1  # ödemesi devam eden (Active) krediler
    print("New Binary Column calculated")

    # We take only positive values of  ENDDATE since we are looking at Bureau Credit VALID IN FUTURE
    # as of the date of the customer's loan application with Home Credit
    B1 = br[br['CREDIT_ENDDATE_BINARY'] == 1]
    del br["CREDIT_ENDDATE_BINARY"]

    # Calculate Difference in successive future end dates of CREDIT
    # Create Dummy Column for CREDIT_ENDDATE
    B1['DAYS_CREDIT_ENDDATE1'] = B1['DAYS_CREDIT_ENDDATE']
    # Groupby Each Customer ID
    grp = B1[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by=['SK_ID_CURR'])
    # Sort the values of CREDIT_ENDDATE for each customer ID
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending=True)).reset_index(drop=True)
    del grp
    gc.collect()
    print("Grouping and Sorting done")

    # Calculate the Difference in ENDDATES and fill missing values with zero
    grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff()
    grp1['NEW_DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
    del grp1['DAYS_CREDIT_ENDDATE1'], grp1['SK_ID_CURR']
    gc.collect()
    print("Difference days calculated")

    # Merge new feature 'DAYS_ENDDATE_DIFF' with original Data frame for BUREAU DATA
    br = br.merge(grp1, on=['SK_ID_BUREAU'], how='left')
    del grp1
    gc.collect()

    # FEATURE 8 - DEBT OVER CREDIT RATIO
    # The Ratio of Total Debt to Total Credit for each Customer
    # A High value may be a red flag indicative of potential default

    br['AMT_CREDIT_SUM_DEBT'] = br['AMT_CREDIT_SUM_DEBT'].fillna(0)
    br['AMT_CREDIT_SUM'] = br['AMT_CREDIT_SUM'].fillna(0)

    grp1 = br[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str,
                                                          columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})

    grp2 = br[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(
        index=str, columns={'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})

    br = br.merge(grp1, on=['SK_ID_CURR'], how='left')
    br = br.merge(grp2, on=['SK_ID_CURR'], how='left')
    del grp1, grp2
    gc.collect()

    br['NEW_DEBT_CREDIT_RATIO'] = br['TOTAL_CUSTOMER_DEBT'] / br['TOTAL_CUSTOMER_CREDIT']

    del br['TOTAL_CUSTOMER_DEBT'], br['TOTAL_CUSTOMER_CREDIT']
    gc.collect()

    # FEATURE 9 - OVERDUE OVER DEBT RATIO
    # What fraction of total Debt is overdue per customer?
    # A high value could indicate a potential DEFAULT

    br['AMT_CREDIT_SUM_DEBT'] = br['AMT_CREDIT_SUM_DEBT'].fillna(0)
    br['AMT_CREDIT_SUM_OVERDUE'] = br['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

    grp1 = br[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str,
                                                          columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    grp2 = br[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(index=str, columns={
        'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})

    br = br.merge(grp1, on=['SK_ID_CURR'], how='left')
    br = br.merge(grp2, on=['SK_ID_CURR'], how='left')
    del grp1, grp2
    gc.collect()

    br['NEW_OVERDUE_DEBT_RATIO'] = br['TOTAL_CUSTOMER_OVERDUE'] / br['TOTAL_CUSTOMER_DEBT']

    del br['TOTAL_CUSTOMER_OVERDUE'], br['TOTAL_CUSTOMER_DEBT']
    gc.collect()

    # FEATURE 10 - AVERAGE NUMBER OF LOANS PROLONGED
    # Müşter
    br['CNT_CREDIT_PROLONG'] = br['CNT_CREDIT_PROLONG'].fillna(0)
    grp = br[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by=['SK_ID_CURR'])[
        'CNT_CREDIT_PROLONG'].mean().reset_index().rename(index=str, columns={
        'CNT_CREDIT_PROLONG': 'NEW_AVG_CREDITDAYS_PROLONGED'})
    br = br.merge(grp, on=['SK_ID_CURR'], how='left')

    # One Hot
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(br)

    # high_correlation(br, remove=['SK_ID_CURR','SK_ID_BUREAU'], corr_coef = "spearman", corr_value = 0.7)

    drop_list_1 = ["NEW_CREDIT_OVERDUE_SUBSTRACT", "NEW_DEBT_SUM_TO_CREDIT_SUM_RATIO", "NEW_PAID_CREDIT_PERC",
                   "NEW_HAS_CREDIT_CARD", "NEW_OVERDUE_CREDIT_SUM_PERC"]
    br.drop(drop_list_1, axis=1, inplace=True)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category=True)
    br, br_cat = one_hot_encoder(br, nan_as_category=True)

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(br)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    br = br.join(bb_agg, how='left', on='SK_ID_BUREAU')
    br.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'SK_ID_CURR': ['count'],
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['min', 'max'],
        'CNT_CREDIT_PROLONG': ['min', 'max'],
        'AMT_CREDIT_SUM': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'min'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean', 'min', 'sum'],
        'NEW_DAYS_CREDIT_UPDATE_SUBSTRACT': ['min', 'max', 'sum', 'mean'],
        'NEW_CREDIT_PAID_SUBSTRACT': ["count", "min", "max", "mean"],
        # 'NEW_OVERDUE_CREDIT_SUM_PERC': ['min', 'mean', 'max'],
        # 'NEW_DEBT_SUM_TO_CREDIT_SUM_RATIO': ['min', 'mean', 'max'],
        'NEW_PAID_CREDIT': ['sum'],
        'NEW_DEBT_CREDIT_RATIO': ['min', 'sum'],
        # 'NEW_PAID_CREDIT_PERC': ['mean'],
        'NEW_CREDIT_OVERDUE_PROLONG_SUM': ['mean'],
        'NEW_CREDIT_OVERDUE_PROLONG_MAX': ['mean', 'min'],
        # 'NEW_CREDIT_OVERDUE_SUBSTRACT': ['mean', 'min', 'max'],
        'NEW_CREDIT_OVERDUE_RATIO': ['min', 'max'],
        # 'NEW_BUREAU_LOAN_COUNT': ["min","max","sum","mean"],
        'NEW_AVERAGE_LOAN_TYPE': ["min", "max"],
        'NEW_BUREAU_LOAN_TYPES': ["min", "max", "mean", "sum"],
        'NEW_ACTIVE_LOANS_PERCENTAGE': ["min", "max"],
        'NEW_DAYS_DIFF': ["max", "mean"],
        'NEW_CREDIT_ENDDATE_PERCENTAGE': ["max"],
        'DAYS_ENDDATE_DIFF': ["min", "max", "mean"],
        'NEW_OVERDUE_DEBT_RATIO': ["max", "mean"],
        'NEW_EARLY': ["sum"],
        'NEW_LATE': ["sum"],
        # 'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']}

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in br_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = br.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # high_correlation(bureau_agg, remove=['SK_ID_CURR','SK_ID_BUREAU'], corr_coef = "spearman", corr_value = 0.7)

    # kisinin aldıgı en yuksek ve en dusuk kredinin farkını gösteren yeni degisken
    bureau_agg["BURO_NEW_AMT_CREDIT_SUM_RANGE"] = bureau_agg["BURO_AMT_CREDIT_SUM_MAX"] - bureau_agg[
        "BURO_AMT_CREDIT_SUM_MIN"]

    # # ortalama kac ayda bir kredi cektigini ifade eden  yeni degisken
    # bureau_agg["BURO_NEW_DAYS_CREDIT_RANGE"]= round((bureau_agg["BURO_DAYS_CREDIT_MAX"] - bureau_agg["BURO_DAYS_CREDIT_MIN"])/(30 * bureau_agg["BURO_SK_ID_CURR_COUNT"]))



    # NEW_EARLY_RATIO
    bureau_agg['NEW_EARLY_RATIO'] = bureau_agg['BURO_NEW_EARLY_SUM'] / bureau_agg[
        'BURO_NEW_CREDIT_PAID_SUBSTRACT_COUNT']  # Erken ödeme oranı

    # NEW_LATE_RATIO
    bureau_agg['NEW_LATE_RATIO'] = bureau_agg['BURO_NEW_LATE_SUM'] / bureau_agg[
        'BURO_NEW_CREDIT_PAID_SUBSTRACT_COUNT']  # Geç ödeme oranı

    # Bureau: Active credits - using only numerical aggregations
    active = br[br['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = br[br['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, br
    gc.collect()
    return bureau_agg
