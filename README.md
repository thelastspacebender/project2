# Receptor Simulator 🧬

*(Russian version below / Русская версия ниже)*

A full-featured Desktop Application for designing, constructing, and training neural network receptor ensembles (MoE).

## Features:
1. **Visual Architecture Constructor**: An interactive graph editor where you can add/remove layers and change the number of neurons using a user-friendly interface (with schematic rendering of neurons and connections).
2. **Feature Management**: Built-in Random Forest algorithm configuration for automatically selecting the best Dragon descriptors from the dataset.
3. **Network Physics Tuning**: Full control over hyperparameters:
   - **L1 (Lasso)** — to "silence" unnecessary receptors and discover concise physical laws.
   - **Weight Decay (L2)** — to prevent overfitting.
   - **Dropout** — for network robustness.
4. **Powerful Ensemble**: Instead of a single network, you configure and launch the training of an entire army (default is 30 intersecting folds — ShuffleSplit Bagging).
5. **Live Dashboard**: Launching the simulation opens a classic interactive Matplotlib dashboard that displays:
   - Live Train/Val Loss and R2 graphs
   - Receptor activity heatmap
   - Fair progression of the entire ensemble (bar chart)
   - Honest final **Out-of-Fold (OOF) R2**.

## Installation and Usage:
1. Clone the repository and navigate to the `dev` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > **Important note about PyTorch:** The `requirements.txt` file specifically contains a command to install the **lightweight CPU-version** of PyTorch (~150 MB instead of 2.5 GB). For our tabular neural network, CPU computations are faster than GPU, so heavy CUDA drivers are not needed.
3. Launch the application: 
   ```bash
   run.bat
   ```
   *(Or run `python main.py` directly)*
4. In the opened window, configure the architecture (click `[+]`, `[-]`, `[+ Layer]`, `[×]`).
5. Click the giant green button **🚀 LAUNCH SIMULATION**.

## Balancing Tips:
- If the dashboard shows that almost all receptors have shut down (e.g., Active: 6/14), and R2 does not grow above 0.75, **weaken the L1 Regularization** (make it `0.00001` instead of `0.0001`). 
- Increasing the number of folds (e.g., to 50) will make the final ensemble more accurate but will increase training time.

---

# Receptor Simulator (Русская версия) 🧬

Полноценное Desktop-приложение для моделирования, конструирования и обучения ансамблей нейросетей-рецепторов (MoE).

## Возможности:
1. **Визуальный конструктор архитектуры**: Интерактивный редактор графов, где вы можете добавлять/удалять слои и изменять количество нейронов с помощью удобного интерфейса (схематичная отрисовка нейронов и связей).
2. **Управление фичами**: Встроенная настройка алгоритма Random Forest для автоматического отбора лучших дескрипторов Dragon из датасета.
3. **Настройка физики сети**: Полный контроль над гиперпараметрами:
   - **L1 (Лассо)** — для "выключения" ненужных рецепторов и поиска лаконичных законов физики.
   - **Weight Decay (L2)** — для борьбы с переобучением.
   - **Dropout** — для устойчивости сети.
4. **Мощнейший Ансамбль**: Вместо одной сети вы настраиваете и запускаете обучение целой армии (по умолчанию 30 фолдов с пересечением — ShuffleSplit Bagging).
5. **Live Dashboard**: При запуске открывается классический интерактивный дашборд Matplotlib, который отображает:
   - Live-графики Train/Val Loss и R2
   - Тепловую карту активности рецепторов
   - Честный прогресс всего ансамбля (бар-чарт)
   - Честный финальный **Out-of-Fold (OOF) R2**.

## Установка и запуск:
1. Клонируйте репозиторий и перейдите в папку `dev`.
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
   > **Важно про PyTorch:** В файле `requirements.txt` специально прописана команда для установки **облегченной CPU-версии** PyTorch (весит около 150 МБ вместо 2.5 ГБ). Для нашей табличной нейросети вычисления на процессоре работают быстрее, чем на видеокарте, поэтому тяжелые CUDA-драйверы не нужны.
3. Запустите приложение: 
   ```bash
   run.bat
   ```
   *(Или `python main.py` напрямую)*
4. В открывшемся окне настройте архитектуру (нажимайте `[+]`, `[-]`, `[+ Layer]`, `[×]`).
5. Нажмите огромную зеленую кнопку **🚀 LAUNCH SIMULATION**.

## Советы по балансу:
- Если дашборд показывает, что почти все рецепторы отключились (например, Active: 6/14), а R2 не растет выше 0.75, **ослабьте L1 Regularization** (сделайте его `0.00001` вместо `0.0001`). 
- Увеличение количества фолдов (например, до 50) сделает финальный ансамбль более точным, но увеличит время обучения.
