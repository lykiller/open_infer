import pandas as pd

if __name__ == "__main__":
    xlsx_path = r"D:\能源消耗数据.xlsx"
    data = pd.read_excel(xlsx_path)

    value_name_list = ["股票", "年份"]
    for value_name in data["ziyuan"]:
        if value_name not in value_name_list:
            value_name_list.append(value_name)

    multi_index_list = []
    for gupiao, year in zip( data["gupiao"], data["year"]):
        if (gupiao, year) not in multi_index_list:
            multi_index_list.append(str(gupiao) + "," + str(year))
    data_dst = pd.DataFrame(columns=value_name_list, index=multi_index_list)

    for year, gupiao,  ziyuan, xiaohaoliang, danwei in zip(data["year"], data["gupiao"], data["ziyuan"], data["xiaohaoliang"], data["danwei"]):
        data_dst.loc[str(gupiao) + "," + str(year), "年份"] = year
        data_dst.loc[str(gupiao) + "," + str(year), "股票"] = gupiao
        if pd.notna(xiaohaoliang):
            data_dst.loc[str(gupiao) + "," + str(year), ziyuan] = str(xiaohaoliang) + str(danwei)

    print(data_dst)

    data_dst.to_excel(r"D:\convert_能源消耗数据.xlsx", index=False)
