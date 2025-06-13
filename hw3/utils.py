import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl


def get_df_info(df, thr=1.0):
    """
    Выводит инфу о колонках датафрейма в виде датафрейма

    df: исходный датафрейм\\
    thr: порог для trash_score (по умолчанию 1, т.к. колонка с единственным значением мусорная)

    returns: pd.DataFrame с инфой
    """
    # your code here

    
    def df_freq_to_float(s):
        """
        Преобразует форматированную строку с частотой в число

        s: str формата "char: float" или "-1"

        returns: float
        """
        if s == "-1":
            return 0.0
        return float(s.split()[-1])


    df_info = pd.DataFrame(
        index=df.columns,
        columns=[
            "data_type",
            "count_unique",
            "example_1",
            "example_2",
            "zero",
            "nan",
            "empty_str",
            "vc_max",
            "vc_max_freq",
            "trash_score",
        ],
    )

    df = df.copy()
    df.replace("", "<empty_str>", inplace=True)

    for col in df.columns:
        df_info.loc[col, "data_type"] = df[col].dtype.name
        df_info.loc[col, "count_unique"] = df[col].nunique(dropna=False)

        # выбор различных случайных примеров
        no_nan_col = df[col].dropna()
        df_info.loc[col, "example_1"], df_info.loc[col, "example_2"] = (
            "<no_example>",
            "<no_example>",
        )
        if (size := no_nan_col.nunique()) == 1:
            df_info.loc[col, "example_1"] = no_nan_col.iloc[0]
        elif size >= 2:
            (
                df_info.loc[col, "example_1"],
                df_info.loc[col, "example_2"],
            ) = np.random.choice(pd.unique(no_nan_col), size=2, replace=False)

        # подсчет нулей
        try:
            tmp = np.isclose(df[col], 0).mean()
            zero_frac = "z: {:.3f}".format(tmp) if ~np.isclose(tmp, 0) else "-1"
        except:
            zero_frac = "-1"
        df_info.loc[col, "zero"] = zero_frac

        # подсчет нанов
        df_info.loc[col, "nan"] = (
            "n: {:.3f}".format(tmp)
            if ~np.isclose(tmp := df[col].isna().mean(), 0)
            else "-1"
        )

        # подсчет пустых строк
        df_info.loc[col, "empty_str"] = (
            "e: {:.3f}".format(tmp)
            if ~np.isclose(tmp := (df[col] == "<empty_str>").mean(), 0)
            else "-1"
        )

        # подсчет самого частовстречающегося элемента
        if size:
            vc_max = no_nan_col.value_counts(normalize=True).sort_values(
                ascending=False
            )
            df_info.loc[col, "vc_max"] = vc_max.index[0]
            df_info.loc[col, "vc_max_freq"] = "{:.3f}".format(vc_max.iloc[0])
        else:
            df_info.loc[col, "vc_max"] = "No vc_max"
            df_info.loc[col, "vc_max_freq"] = "0"

        # вычисление <<мусорности>> колонки
        zero, nan, empty_str, vc_max_freq = (
            df_freq_to_float(df_info.loc[col, "zero"]),
            df_freq_to_float(df_info.loc[col, "nan"]),
            df_freq_to_float(df_info.loc[col, "empty_str"]),
            df_freq_to_float(df_info.loc[col, "vc_max_freq"]),
        )
        trash_score = max(
            zero + nan + empty_str, vc_max_freq if vc_max_freq > thr else 0
        )
        df_info.loc[col, "trash_score"] = (
            "{:.3f}".format(trash_score) if ~np.isclose(trash_score, 0) else "-1"
        )

    return df_info.sort_values(by="trash_score", ascending=False)


def plot_density(df, hue, cols=None, drop_zero=False, max_cat_thr=20, bins=None):
    """
    Рисует распределения колонок cols

    df: отрисовываемый датафрейм\\
    hue: колонка, по которой бьётся раскраска\\
    cols: отрисовываемые колонки. Если None, то рисуем df.columns (кроме hue)\\
    drop_zero: если True, то выкидываем нули из числовых колонок\\
    max_cat_thr: порог количества значений категориальных признаков\\
    bins: количество бинов для построения гистограмм; если None, то bins='auto'
    """
    # your code here
    if cols is None:
        cols = df.columns

    # определяем, какие признаки числовые, а какие категориальные
    cat_feat, num_feat = [], []
    for col in cols:
        if col != hue:
            if df[col].nunique() <= max_cat_thr:
                cat_feat.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                num_feat.append(col)

    df_copy = df.copy()
    df_copy.fillna(value={hue: "<NaN>"}, inplace=True)
    df_copy.replace(to_replace="", value={hue: "<empty_str>"}, inplace=True)

    # отрисовка числовых признаков
    for col in num_feat:
        fig, ax = plt.subplots(1, 3)
        fig.suptitle(col + " vs. " + hue, fontsize=15)
        temp_df = df_copy.dropna()

        # drop zeros
        if drop_zero:
            temp_df = df_copy[~np.isclose(df_copy[col], 0)]
        hue_order = sorted(pd.unique(temp_df[hue]))

        # hist
        if bins is None:
            bins = "auto"

        sns.histplot(
            data=temp_df,
            x=col,
            hue=hue,
            hue_order=hue_order,
            bins=bins,
            multiple="stack",
            element="step",
            stat="count",
            alpha=0.8,
            ax=ax[0],
            legend=False,
        )
        ax[0].set_xlabel(None)
        ax[0].set_ylabel(None)
        ax[0].set_title("drop_zero=" + str(drop_zero), fontsize=10)

        # special values (0, nan)
        # конструируем датафрейм
        s_df = df_copy[[col, hue]].copy()

        nans = s_df.groupby([hue], as_index=False)[col].apply(lambda x: x.isna().mean())
        nans.rename(columns={col: "value"}, inplace=True)
        nans = pd.DataFrame(nans)
        nans["cat"] = "<NaN>"

        s_df[col] = np.isclose(s_df[col], 0)
        zeros = s_df[[col, hue]].groupby([hue], as_index=False).mean()
        zeros.rename(columns={col: "value"}, inplace=True)
        zeros["cat"] = "0"

        s_df = pd.concat((nans, zeros), axis=0)
        s_df.loc[np.isclose(s_df["value"], 0), "value"] = -0.1 * s_df["value"].max()

        # отображаем датафрейм через barplot
        sns.barplot(
            data=s_df,
            x="cat",
            y="value",
            order=["0", "<NaN>"],
            hue=hue,
            hue_order=hue_order,
            legend=False,
            edgecolor="black",
            ax=ax[1],
        )
        ax[1].grid(True, axis="y")
        ax[1].axhline(0, color="black", ls="--")
        ax[1].tick_params("x", rotation=90, labelsize=10)
        ax[1].set_xlabel(None)
        ax[1].set_ylabel(None)
        ax[1].set_title("special values", fontsize=10)

        # boxen & strip
        sns.boxenplot(
            data=temp_df,
            y=col,
            hue=hue,
            hue_order=hue_order,
            showfliers=False,
            ax=ax[2],
            legend=False,
        )
        sns.stripplot(
            data=temp_df.sample(n=min(temp_df.shape[0], 200)),
            y=col,
            hue=hue,
            hue_order=hue_order,
            ax=ax[2],
            palette="dark:black",
            dodge=True,
            legend=False,
        )
        ax[2].set_xlabel(None)
        ax[2].set_ylabel(None)
        ax[2].set_title("no outliers + sampled stripplot", fontsize=10)

        fig.set_size_inches(12 + 2 * len(hue_order), 4)
        plt.show()

    # отрисовка категориальных признаков
    for col in cat_feat:
        fig, ax = plt.subplots()
        fig.suptitle(col + " vs. " + hue, fontsize=15)

        df_copy.replace("", "<empty_str>", inplace=True)
        df_copy.fillna("<NaN>", inplace=True)

        # countplot
        sns.countplot(
            data=df_copy,
            x=col,
            hue=hue,
            hue_order=hue_order,
            stat="count",
            edgecolor="black",
            ax=ax,
            legend=False
        )
        ax.set_xlabel(None)
        ax.tick_params(axis="x", rotation=90, labelsize=10)
        ax.grid(True, axis="y")

        fig.set_size_inches(8 + 2 * len(hue_order), 4)
        plt.show()

    return


def extract_attr_by_leaf_matrix(t, leaf_matrix, attr_name):
    t_prep = t.copy()
    t_prep['leaf_id'] = t_prep.node_index.str.split('-').str.get(1).str.slice(1, 10).astype(int)
    t_prep = t_prep.query('split_gain.isnull()').pivot_table(values=attr_name, index='tree_index', columns='leaf_id')
    res = []
    for i in range(t_prep.shape[0]): # кол-во деревьев
        res.append(t_prep.iloc[i][leaf_matrix[:, i]])
    res = np.array(res).T
    return res


def my_beeswarm(
    df: pd.DataFrame,
    features: list,
    shap_values: pd.DataFrame,
    cat_feature_threshold: float,
    top_k: int,
    figsize: tuple,
    dots: int,
    alpha_outliers: float = 0.1,
    random_seed: int = None,
):
    """
    Аналог shap.plots.beeswarm с адекватной отрисовкой категориальных признаков и пропущенных значений.

    Args:
        df (pd.DataFrame): Исходный датасет.
        features (list): Список признаков.
        shap_values (pd.DataFrame): Датафрейм с SHAP-значениями для модели LightGBM.
        cat_feature_threshold (float): Порог для определения достаточно "популярных" категорий.
            Категории с долей <= `cat_feature_threshold` не отрисовываются.
        top_k (int): Количество признаков для отрисовки.
        figsize (tuple): Размер графика.
        dots (int): Количество точек, отображаемых для каждого признака.
        alpha_outliers (float): Уровень квантилей для отсечения выбросов при нормализации цветовой шкалы.
            Берутся квантили уровней `alpha_outliers` и `1 - alpha_outliers` соответственно.
            По умолчанию: `0.1`.
        random_seed (int): random_seed для воспроизводимости сэмплинга точек. По умолчанию: `None`.


    Notes:
        - Для категориальных признаков используется зеленая заливка.
        - Пропущенные значения признаков (`NaN`) отображаются черными точками.
        - Цветовая шкала для числовых признаков нормализуется через `AsinhNorm`.
    """
    df = df[features].copy()
    random_state = np.random.RandomState(random_seed)

    # определение категориальных и числовых признаков
    # + вычисление квантилей для нормализации отрисовки числовых признаков
    num_features, cat_features = [], []
    num_feat_quantiles = {}
    for feature in features:
        if isinstance(df[feature].dtype, pd.CategoricalDtype):
            cat_features.append(feature)
        else:
            num_features.append(feature)
            tmp = df[feature]
            num_feat_quantiles[feature] = (
                tmp.quantile(q=alpha_outliers),
                tmp.quantile(q=1 - alpha_outliers),
            )

    # отбрасывание последней колонки в шапах
    shap_df = pd.DataFrame(shap_values[:, :-1], columns=features)

    # создание dummy фичей для "популярных" категорий
    new_dummy_feats = []
    for feature in cat_features:
        vc = df[feature].value_counts(normalize=True)
        new_columns, shap_new_cols = {}, {}
        for feat_value in vc.keys():
            if vc[feat_value] > cat_feature_threshold:
                new_feat_name = f"{feature}_{feat_value}"
                new_dummy_feats.append(new_feat_name)
                idx = df[feature] == feat_value
                new_columns[new_feat_name] = idx.astype(int)
                shap_new_cols[new_feat_name] = shap_df[feature][idx]

        df = pd.concat(
            [df.drop(feature, axis=1), pd.DataFrame(new_columns)], axis=1
        )

        shap_df = pd.concat(
            [shap_df.drop(feature, axis=1), pd.DataFrame(shap_new_cols)],
            axis=1,
        )

    # сортировка признаков по убыванию среднего модуля шапов
    total_sorted_features = (
        abs(shap_df)
        .mean(axis=0)
        .sort_values(ascending=False)
        .index.to_list()[:top_k]
    )

    # подготовка данных к отрисовке
    plot_data = pd.concat(
        [
            shap_df[total_sorted_features].melt(
                var_name="feature", value_name="shap_value"
            ),
            df[total_sorted_features].melt(
                var_name="feature", value_name="feature_value"
            )["feature_value"],
        ],
        axis=1,
    )

    _, ax = plt.subplots(figsize=figsize)

    for i, feature in enumerate(total_sorted_features):
        ax.axhline(y=i, color="gray", alpha=0.25)

        # в dummy признаках если значение 0, то SHAP полагается равным NaN,
        # поэтому для корректной отрисовки необходимого числа точек нужно проворачивать следующий трюк:
        cur_plot = plot_data[
            (plot_data["feature"] == feature)
            & ~(plot_data["shap_value"].isna())
        ]
        cur_plot = cur_plot.sample(
            n=min(dots, cur_plot.shape[0]),
            replace=False,
            random_state=random_state,
        )

        # отрисовка кат. признаков
        if feature in new_dummy_feats:
            sns.stripplot(
                data=cur_plot,
                x="shap_value",
                y="feature",
                ax=ax,
                jitter=0.1,
                legend=False,
                color="green",
                alpha=0.5,
                size=3.5,
            )
        else:
            vmin, vmax = num_feat_quantiles[feature]
            nan_idx = cur_plot["feature_value"].isna()
            # отрисовка числовых признаков
            sns.stripplot(
                data=cur_plot[~nan_idx],
                x="shap_value",
                y="feature",
                ax=ax,
                hue="feature_value",
                hue_norm=mpl.colors.AsinhNorm(
                    vmin=vmin, vmax=vmax, clip=True
                ),  # нормализация цветовой шкалы
                palette="coolwarm",
                jitter=0.2,
                legend=False,
                alpha=0.75,
                size=3.5,
            )
            # отрисовка пропущенных значений
            sns.stripplot(
                data=cur_plot[nan_idx],
                x="shap_value",
                y="feature",
                ax=ax,
                color="black",
                jitter=0.05,
                legend=False,
                alpha=1,
                size=3.5,
            )
    # добавление цветовой шкалы в легенде
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=mpl.colors.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    ax.set_ylabel("Features")
    ax.set_xlabel("SHAP values")
    ax.axvline(x=0)
    plt.show()
