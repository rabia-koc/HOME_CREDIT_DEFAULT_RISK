
def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # FEATURE_1: NAME_TYPE_SUITE

    df.loc[(df["NAME_TYPE_SUITE"] == "Other_B"), "NAME_TYPE_SUITE"] = 1
    df.loc[(df["NAME_TYPE_SUITE"] == "Other_A"), "NAME_TYPE_SUITE"] = 2
    df.loc[(df["NAME_TYPE_SUITE"] == "Group of people"), "NAME_TYPE_SUITE"] = 2
    df.loc[(df["NAME_TYPE_SUITE"] == "Unaccompanied"), "NAME_TYPE_SUITE"] = 2
    df.loc[(df["NAME_TYPE_SUITE"] == "Spouse, partner"), "NAME_TYPE_SUITE"] = 3
    df.loc[(df["NAME_TYPE_SUITE"] == "Children"), "NAME_TYPE_SUITE"] = 3
    df.loc[(df["NAME_TYPE_SUITE"] == "Family"), "NAME_TYPE_SUITE"] = 3

    df["NAME_TYPE_SUITE"].unique()

    # FEATURE_2: NAME_EDUCATION_TYPE

    edu_map = {'Lower secondary': 1,
               'Secondary / secondary special': 2,
               'Incomplete higher': 2,
               'Higher education': 3,
               'Academic degree': 4}

    df["NAME_EDUCATION_TYPE"].unique()

    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(edu_map).astype('int')

    # FEATURE_3: NAME_FAMILY_STATUS

    fam_map = {'Civil marriage': 1,
               'Single / not married': 1,
               'Separated': 2,
               'Married': 3,
               'Widow': 4,
               'Unknown': 5}

    df["NAME_FAMILY_STATUS"].unique()

    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].map(fam_map).astype('int')

    # FEATURE_4: NAME_HOUSING_TYPE

    df.loc[(df["NAME_HOUSING_TYPE"] == "Rented apartment"), "NAME_HOUSING_TYPE"] = 1
    df.loc[(df["NAME_HOUSING_TYPE"] == "With parents"), "NAME_HOUSING_TYPE"] = 2
    df.loc[(df["NAME_HOUSING_TYPE"] == "Municipal apartment"), "NAME_HOUSING_TYPE"] = 3
    df.loc[(df["NAME_HOUSING_TYPE"] == "Co-op apartment"), "NAME_HOUSING_TYPE"] = 4
    df.loc[(df["NAME_HOUSING_TYPE"] == "House / apartment"), "NAME_HOUSING_TYPE"] = 4
    df.loc[(df["NAME_HOUSING_TYPE"] == "Office apartment"), "NAME_HOUSING_TYPE"] = 5

    df["NAME_HOUSING_TYPE"].unique()

    df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].astype('int')

    # FEATURE_5: OCCUPATION_TYPE

    df.loc[(df["OCCUPATION_TYPE"] == "Low-skill Laborers"), "OCCUPATION_TYPE"] = 1
    df.loc[(df["OCCUPATION_TYPE"] == "Drivers"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Waiters/barmen staff"), "OCCUPATION_TYPE"] = 2
    df.loc[(df["OCCUPATION_TYPE"] == "Security staff"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Laborers"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Cooking staff"), "OCCUPATION_TYPE"] = 3
    df.loc[(df["OCCUPATION_TYPE"] == "Sales staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Cleaning staff"), "OCCUPATION_TYPE"] = 4
    df.loc[(df["OCCUPATION_TYPE"] == "Realty agents"), "OCCUPATION_TYPE"] = 5
    df.loc[(df["OCCUPATION_TYPE"] == "Secretaries"), "OCCUPATION_TYPE"] = 6
    df.loc[(df["OCCUPATION_TYPE"] == "Medicine staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "Private service staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "IT staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "HR staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "Core staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "High skill tech staff"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "Managers"), "OCCUPATION_TYPE"] = 7
    df.loc[(df["OCCUPATION_TYPE"] == "Accountants"), "OCCUPATION_TYPE"] = 8

    df["OCCUPATION_TYPE"].unique()

    # FEATURE_6: REGION_RATING_CLIENT

    rate_map = {3: 1,
                2: 2,
                1: 3}

    df['REGION_RATING_CLIENT'] = df['REGION_RATING_CLIENT'].map(rate_map)

    df["REGION_RATING_CLIENT"].unique()

    # FEATURE_7: Kişinin başvurudan önceki çalışma gün sayısının kişinin yaşının gün cinsine oranı
    df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # FEATURE_8: Müşterinin gelirinin kredi tutarına oranı
    df['NEW_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

    # FEATURE_9: Müşterinin gelirinin aile üyesi miktarına oranı (Kişi başı gelir)
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # FEATURE_10: Kredi yıllık ödemesinin müşterinin gelirine oranı
    df['NEW_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # FEATURE_11: Kredi yıllık ödemesinin toplam kredi tutarına oranı
    df['NEW_PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # FEATURE_12: Toplam kredi tutarının malların tutarına oranı
    df["NEW_GOODS_RATE"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

    # FEATURE_13: Malın fiyatını geçen kredi analizi
    df['NEW_OVER_EXPECT_CREDIT'] = (df.AMT_CREDIT > df.AMT_GOODS_PRICE).replace({False: 0, True: 1})

    # FEATURE_14: Müşterinin gelirinin çocuk sayısına oranı (Saçma, düşükse sil)
    df['NEW_INCOME_FOR_CHILD_RATE'] = [x / y if y != 0 else 0 for x, y in
                                       df[['AMT_INCOME_TOTAL', 'CNT_CHILDREN']].values]

    # FEATURE_15: Aile Üyeleri - Çocuk Sayısı = Yetişkin Sayısı
    df["NEW_CNT_ADULTS"] = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]

    # FEATURE_16: Çocuk sayısının aile üyesi miktarına oranı
    df['NEW_CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    # FEATURE_17: Müşterinin yaşı (yıl cinsinden)
    df['NEW_DAYS_BIRTH'] = df['DAYS_BIRTH'] * -1 / 365
    df["NEW_YEARS_BIRTH"] = round(df['NEW_DAYS_BIRTH'], 0)

    # FEATURE_18: Sağlanan belgelerin toplamı
    doc_list = ["FLAG_DOCUMENT_3", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6",
                "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_11",
                "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
                "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19"]

    df["NEW_SUM_DOCUMENTS"] = df[doc_list].sum(axis=1)

    # FEATURE_19: Hafta içi - hafta sonu sınıflandırması
    df["NEW_DAY_APPR_PROCESS_START"] = "Weekdays"
    df["NEW_DAY_APPR_PROCESS_START"][
        (df["WEEKDAY_APPR_PROCESS_START"] == "SATURDAY") | (df["WEEKDAY_APPR_PROCESS_START"] == "SUNDAY")] = "Weekend"

    # df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)  # Araba yaş değeri boş olan gözlemler 0 olarak atandı

    # FEATURE_20: Müşterinin arabasının yaşının müşterinin günlük yaşına oranı
    df["NEW_OWN_CAR_AGE_PERC"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]

    # FEATURE_21: Kişinin araba yaşının çalıştığı yıla oranı
    df['NEW_CAR_EMPLOYED_PERC'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']

    # FEATURE_22: Müşterinin yaşınınmüşterinin arabasının yaşına oranı
    df["NEW_YEARS_CAR_PERC"] = df["NEW_YEARS_BIRTH"] / df["OWN_CAR_AGE"]

    # df["DAYS_ID_PUBLISHED_RATIO"] = df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"]
    # df["DAYS_REGISTRATION_RATIO"] = df["DAYS_REGISTRATION"] / df["DAYS_BIRTH"]
    # df["DAYS_LAST_PHONE_CHANGE_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]

    # FEATURE_23: Müşterinin yaşadığı yerin ortalama değerlendirmesinin toplamı
    living_avg_list = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', \
                       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', \
                       'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', \
                       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']

    df["NEW_SUM_LIVING_AVG"] = df[living_avg_list].sum(axis=1)

    # FEATURE_24: Müşterinin yaşadığı yerin mode değerlendirmesinin toplamı
    living_mode_list = ['APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                        'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
                        'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
                        'NONLIVINGAREA_MODE', 'TOTALAREA_MODE']

    df["NEW_SUM_LIVING_MODE"] = df[living_mode_list].sum(axis=1)

    # FEATURE_25: Dışarıdan alınan normalleştirilmiş puanların ortalaması
    df['NEW_EXT_SOURCE_MEAN'] = (df['EXT_SOURCE_1'] +
                                 df['EXT_SOURCE_2'] +
                                 df['EXT_SOURCE_3']) / 3

    # FEATURE_26: Dışarıdan alınan normalleştirilmiş puanların çarpımı
    df['NEW_EXT_SOURCE_MUL'] = df['EXT_SOURCE_1'] * \
                               df['EXT_SOURCE_2'] * \
                               df['EXT_SOURCE_3']

    # FEATURE_27: Dışarıdan alınan normalleştirilmiş puanların varyansı
    df['NEW_EX3_SOURCE_VAR'] = [np.var([ext1, ext2, ext3]) for ext1, ext2, ext3 in
                                zip(df['EXT_SOURCE_1'], df['EXT_SOURCE_2'], df['EXT_SOURCE_3'])]
    # REGION_RATING_CLIENT_W_CITY

    # DEF_60_CNT_SOCIAL_CIRCLE

    ##########################################################################

    # Generating new fetures with using other features

    # FEATURE_28: Dışarıdan alınan normalleştirilmiş puanların toplamı
    df['EXT_SOURCE_SUM'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)

    # FEATURE_29: Kişinin yıl cinsinden yaşının kredi tutarına oranı
    df['BIRTH_VS_CREDIT'] = [(a / b) if b != 0 else 0 for a, b in df[['NEW_YEARS_BIRTH', 'AMT_CREDIT']].values]

    # FEATURE_30: Müşterinin yıl cinsinden yaşının maaşına oranı (Müşterinin yaşam kalitesi) (sorun yaratırsa positive yap)
    df['NEW_BIRTH_INCOME_PERC'] = [(a / b) if b != 0 else 0 for a, b in
                                   df[['NEW_YEARS_BIRTH', 'AMT_INCOME_TOTAL']].values]

    # FEATURE_31: OBS TOTAL
    df["NEW_OBS_30_60"] = df[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']].sum(axis=1)

    # FEATURE_32: DEF TOTAL
    df["NEW_DEF_30_60"] = df[['DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']].sum(axis=1)

    # FEATURE_33: CITY ADRES UYUŞMAZLIKLARI (DAYS_ID_PUBLISH İLE) (DAYS_REGISTRATION İLE DENE)
    df["NEW_CHEAT_CITY"] = df["REG_CITY_NOT_LIVE_CITY"] + \
                           df["REG_CITY_NOT_WORK_CITY"] + \
                           df["LIVE_CITY_NOT_WORK_CITY"]

    df["NEW_CHEAT_CITY_EQ"] = (df["NEW_CHEAT_CITY"] + 1) * \
                              (df["DAYS_ID_PUBLISH"])

    df.drop("NEW_CHEAT_CITY", axis=1, inplace=True)

    # FEATURE_34: REGION ADRES UYUŞMAZLIKLARI (DAYS_ID_PUBLISH İLE) (DAYS_REGISTRATION İLE DENE)
    df["NEW_CHEAT_REGION"] = df["REG_REGION_NOT_LIVE_REGION"] + df["REG_REGION_NOT_WORK_REGION"] + df[
        "LIVE_REGION_NOT_WORK_REGION"]

    df["NEW_CHEAT_REGION_EQ"] = (df["NEW_CHEAT_REGION"] + 1) * \
                                (df["DAYS_ID_PUBLISH"])

    df.drop("NEW_CHEAT_REGION", axis=1, inplace=True)

    # FEATURE_35: Müşterinin yaşadığı bölge için ve müşterinin yaşadığı şehir için puanların çarpımı
    df["NEW_RATING_CLIENT"] = df["REGION_RATING_CLIENT"] * \
                              df["REGION_RATING_CLIENT_W_CITY"]

    # FEATURE_36: Müşterinin yaşadığı bölge ve şehir puanlamasının müşterinin geliri ile çarpılması
    df['NEW_RATING_CLIENT_INCOME'] = (df['REGION_RATING_CLIENT'] + df['REGION_RATING_CLIENT_W_CITY']) * df[
        'AMT_INCOME_TOTAL']

    # FEATURE_37: Kredi tutarının aile yetişkinleri sayısına bölünmesi
    df['NEW_AMT/FAM'] = df['AMT_CREDIT'] / df["NEW_CNT_ADULTS"]

    # FEATURE_38: Kişinin aylık gelir hesabı
    df['NEW_INCOME_IN_A_MONTH'] = df['AMT_INCOME_TOTAL'] / 12

    # FEATURE_38: Kişinin aylık kredi ödeme hesabı
    df['NEW_AMT_ANNUITY_IN_A_MONTH'] = df['AMT_ANNUITY'] / 12

    # FEATURE_39: Kişinin aylık gelirinden, aylık kredi ödeme tutarının çıkarılması (Aylık cepte kalan para)
    df['NEW_MONEY_MONTH'] = df['NEW_INCOME_IN_A_MONTH'] - df['NEW_AMT_ANNUITY_IN_A_MONTH']

    # FEATURE_40: Kredi ödemesinin kaç yılda biteceğinin hesabı
    df["NEW_HOW_MANY_YEARS_CREDIT"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    # FEATURE_41: Kredi tutarının malin fiyatına göre fazlalığı
    df["NEW_CREDIT_GOODS_SUBSTRACT"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]

    # FEATURE_42: Malın fiyatının kredi tutarına göre fazlalığı
    df["NEW_GOODS_CREDIT_SUBSTRACT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]

    # FEATURE_43: Cepte kalan paranın kişi başına düşen miktarı
    df['NEW_MONEY_MONTH_PER_PERSON'] = df['NEW_MONEY_MONTH'] / df['CNT_FAM_MEMBERS']

    # FEATURE_44: Eğitim kırılımı bazında maaş değişkeni oluşturulması
    NEW_INC_EDU = df[['AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE']].groupby('NAME_EDUCATION_TYPE').median()[
        'AMT_INCOME_TOTAL']

    df['NEW_INC_EDU'] = df['NAME_EDUCATION_TYPE'].map(NEW_INC_EDU)

    # FEATURE_45: Meslek kırılımı bazında maaş değişkeni oluşturulması
    NEW_INC_ORG = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()[
        'AMT_INCOME_TOTAL']

    df['NEW_INC_ORG'] = df['ORGANIZATION_TYPE'].map(NEW_INC_ORG)

    # FEATURE_46: Sorguların toplamı
    df['NEW_REQ_CREDIT_BUREAU_SUM'] = df[['AMT_REQ_CREDIT_BUREAU_DAY',
                                          'AMT_REQ_CREDIT_BUREAU_HOUR',
                                          'AMT_REQ_CREDIT_BUREAU_WEEK',
                                          'AMT_REQ_CREDIT_BUREAU_MON',
                                          'AMT_REQ_CREDIT_BUREAU_QRT',
                                          'AMT_REQ_CREDIT_BUREAU_YEAR']].sum(axis=1)

    # FEATURE_47: OBS'de 30 gün ve 60 gün arasına düşen kişiler
    df['NEW_OBS_30_OBS_60_BETWEEN'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] - df['OBS_60_CNT_SOCIAL_CIRCLE']

    # FEATURE_48: DEF'de 30 gün ve 60 gün arasına düşen kişiler
    df['NEW_DEF_30_DEF_60_BETWEEN'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] - df['DEF_60_CNT_SOCIAL_CIRCLE']

    # FEATURE_49: Mal fiyatının evin alan ortalamasına bölünmesi (metrekare başına düşen ücret)
    df['RATIO_AMT_GOODS_PRICE_TO_LIVINGAREA_AVG'] = df['AMT_GOODS_PRICE'] / df['LIVINGAREA_AVG']

    # FEATURE_50: Mal fiyatının binanın yaş ortalamasına bölünmesi
    df['RATIO_AMT_GOODS_PRICE_TO_YEARS_BUILD_AVG'] = df['AMT_GOODS_PRICE'] / df['YEARS_BUILD_AVG']

    # FEATURE_51: Mal fiyatının yaşanılan apartmanın durumuna bölünmesi
    df['RATIO_AMT_GOODS_PRICE_TO_LIVINGAPARTMENTS_AVG'] = df['AMT_GOODS_PRICE'] / df['LIVINGAPARTMENTS_AVG']

    # FEATURE_52: Kaç gün önce telefon değiştirme bilgisinin kaç gün önce kayıt değiştirme bilgisine oranı
    df['RATIO_DAYS_LAST_PHONE_CHANGE_TO_DAYS_REGISTRATION'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_REGISTRATION']

    # FEATURE_53: 1 saat önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_HOUR'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_54: 1 gün önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_DAY'] = df['AMT_REQ_CREDIT_BUREAU_DAY'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_55: 1 hafta önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_WEEK'] = df['AMT_REQ_CREDIT_BUREAU_WEEK'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_56: 1 ay önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_MON'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_57: 3 ay önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_QRT'] = df['AMT_REQ_CREDIT_BUREAU_QRT'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_58: 1 yıl önceki sorgunun tüm sorgulara oranı
    df['PERC_ENQUIRIES_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_YEAR'] / df['NEW_REQ_CREDIT_BUREAU_SUM']

    # FEATURE_59: Normalleştirilmiş puanın max'ı
    df['NEW_EXT_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)

    # FEATURE_60: Normalleştirilmiş puanın min'ı
    df['NEW_EXT_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)

    # FEATURE_61: Flag değişkenlerinin toplanması
    df['NEW_FLAG_CONTACTS_SUM'] = df['FLAG_MOBIL'] + \
                                  df['FLAG_EMP_PHONE'] + \
                                  df['FLAG_WORK_PHONE'] + \
                                  df['FLAG_CONT_MOBILE'] + \
                                  df['FLAG_PHONE'] + \
                                  df['FLAG_EMAIL']

    drop_list_1 = ["NONLIVINGAPARTMENTS_AVG", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAPARTMENTS_MEDI",
                   "AMT_REQ_CREDIT_BUREAU_QRT", "NEW_INCOME_CREDIT_PERC", "NEW_RATING_CLIENT_INCOME",
                   "NEW_INCOME_IN_A_MONTH", "NEW_MONEY_MONTH", "NEW_REQ_CREDIT_BUREAU_SUM",
                   "RATIO_AMT_GOODS_PRICE_TO_YEARS_BUILD_AVG", "RATIO_DAYS_LAST_PHONE_CHANGE_TO_DAYS_REGISTRATION",
                   "PERC_ENQUIRIES_HOUR", "PERC_ENQUIRIES_DAY", "PERC_ENQUIRIES_WEEK"]

    df.drop(drop_list_1, axis=1, inplace=True)

    drop_list_2 = ["APARTMENTS_MODE", "LIVINGAPARTMENTS_MEDI", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_AVG",
                   "LIVINGAREA_MEDI", "BASEMENTAREA_MEDI", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MEDI",
                   "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MEDI", "YEARS_BUILD_MODE", "COMMONAREA_MEDI",
                   "COMMONAREA_MODE", "ELEVATORS_MEDI", "ELEVATORS_MODE", "ENTRANCES_MEDI", "ENTRANCES_MODE",
                   "FLOORSMAX_MEDI", "FLOORSMAX_MODE", "FLOORSMIN_MEDI", "FLOORSMIN_MODE", "LANDAREA_MEDI",
                   "LANDAREA_MODE", "APARTMENTS_AVG", "NONLIVINGAREA_MEDI", "NONLIVINGAREA_MODE", "BASEMENTAREA_MEDI",
                   "BASEMENTAREA_AVG", "YEARS_BUILD_MEDI", "YEARS_BUILD_AVG", "ENTRANCES_MEDI", "ENTRANCES_AVG"]

    df.drop(drop_list_2, axis=1, inplace=True)

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    df.columns = pd.Index(["APP_" + col for col in df.columns.tolist()])
    df.rename(columns={"APP_SK_ID_CURR": "SK_ID_CURR"}, inplace=True)
    df.rename(columns={"APP_TARGET": "TARGET"}, inplace=True)

    del test_df
    gc.collect()
    return df
