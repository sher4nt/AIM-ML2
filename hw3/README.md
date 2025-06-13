# Бустинг на анонимных данных

Данные взяты [отсюда](https://www.kaggle.com/competitions/allstate-claims-severity/overview). Требуется предсказать непрерывный таргет. В задаче в качестве метрики используется эпсилон-чувствительный асимметричный лосс:
```math
    L(y, \hat{y}) = \begin{cases}
        0, & \text{если}\ |y - \hat{y}| \leqslant \varepsilon\\
        \alpha \cdot \left(|y - \hat{y}| - \varepsilon\right)^p, & \text{если}\ \hat{y} > y + \varepsilon \\
        \beta \cdot \left(|y - \hat{y}| - \varepsilon\right)^p, & \text{если}\ \hat{y} < y - \varepsilon
    \end{cases}
```
где $\alpha = 0.9$, $\beta = 1.1$, $p = 2$. Работа включает в себя отбор и генерацию признаков, понижение размерности, интерпретацию модели, реализации кастомных лосс-функций и метрик качества и оптимизацию гиперпараметров. Основная работа проделана в ноутбуке `final.ipynb`. 
