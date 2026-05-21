# RL Lab. От среды OpenAI Gym до глубоких Q-сетей

> Лабораторная работа по дисциплине «интеллектуальные системы». ИжГТУ им. М. Т. Калашникова, 2025.
>
> Автор: **Р. В. Скороходов**, гр. М25-787-1.

Работа состоит из пяти ноутбуков, выстроенных как последовательное знакомство с обучением с подкреплением: от стандартного интерфейса среды до глубоких Q-сетей. Использован пакет `gymnasium` (актуальный поддерживаемый преемник OpenAI Gym). Ноутбуки 01, 02, 05 работают с `CartPole-v1` (непрерывное состояние); ноутбуки 03 и 04 решают одну и ту же задачу `FrozenLake-v1` двумя методами (модельным и модельно-свободным) для прямого сравнения.

> Все Colab-кнопки настроены на репозиторий `megusto0/rl-lab` и работают сразу после клика.

---

## Состав работы

| № | Тема | Ноутбук | Colab |
|---|------|---------|-------|
| 01 | OpenAI Gym и случайный агент | [`01_openai_gym.ipynb`](./01_openai_gym.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/01_openai_gym.ipynb) |
| 02 | Метод кросс-энтропии (CEM) | [`02_cross_entropy.ipynb`](./02_cross_entropy.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/02_cross_entropy.ipynb) |
| 03 | Динамическое программирование (Value Iteration) | [`03_value_iteration.ipynb`](./03_value_iteration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/03_value_iteration.ipynb) |
| 04 | Tabular Q-learning | [`04_q_learning.ipynb`](./04_q_learning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/04_q_learning.ipynb) |
| 05 | Deep Q-Network (DQN) | [`05_dqn.ipynb`](./05_dqn.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/05_dqn.ipynb) |

---

## Описание ноутбуков

### 01. OpenAI Gym и случайный агент
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/01_openai_gym.ipynb)

Знакомство с API библиотеки Gymnasium: создание среды CartPole-v1, цикл взаимодействия `reset → step`, пространства наблюдения и действия. Запускаются случайный агент и простая эвристика на 200 эпизодах для сравнения базовой линии с достижимым максимумом среды. Финальная ячейка сохраняет `results/01_random_baseline.csv` (статистика по эпизодам). Эта таблица копируется в **таблицу 2** отчёта.

**Источник:** Lapan M., *Deep Reinforcement Learning Hands-On*, главы 2, 3.

---

### 02. Метод кросс-энтропии
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/02_cross_entropy.ipynb)

CEM на CartPole-v1. Политика: сеть 4-128-2 с ReLU и Softmax. Гиперпараметры: batch_size = 16 эпизодов, percentile = 70 (верхние 30 % как элита). Останов: средняя награда по батчу ≥ 199. Финальная ячейка сохраняет `results/02_cem_training.csv` с динамикой обучения. Эта таблица копируется в **таблицу 3** отчёта.

**Источник:** Lapan M., глава 4.

---

### 03. Динамическое программирование (Value Iteration)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/03_value_iteration.ipynb)

Value Iteration на стохастической `FrozenLake-v1` (4×4, slippery=True). Модель среды берётся напрямую из `env.unwrapped.P`. Сходимость по критерию `max |ΔV| < 1e-8`, γ = 0,99. Извлекается оптимальная политика и тестируется на 1000 эпизодов. Финальная ячейка сохраняет `results/03_value_iteration.csv`. Соответствует **таблицам 4 и 4-А** отчёта.

**Источники:** Lapan M., глава 5; Sutton & Barto, глава 4.

---

### 04. Tabular Q-learning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/04_q_learning.ipynb)

Модельно-свободный Q-learning на той же `FrozenLake-v1` для прямого сравнения с ноутбуком 03. Гиперпараметры подобраны для устойчивой учебной демонстрации на стохастической среде: γ = 0,99, ε-жадная политика с убыванием ε от 1,0 до 0,01 за 10 000 эпизодов, learning rate убывает от 0,5 до 0,05, всего 50 000 эпизодов обучения. Финальная ячейка сохраняет `results/04_q_learning.csv` со сравнительной таблицей. Соответствует **таблице 5** отчёта.

**Источники:** Lapan M., глава 5; Sutton & Barto, глава 6.

---

### 05. Deep Q-Network (DQN)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/megusto0/rl-lab/blob/main/05_dqn.ipynb)

DQN на CartPole-v1. Q-функция: сеть 4-128-128-2 (ReLU). Replay buffer 10 000, target network с синхронизацией раз в 100 шагов. Оптимизация: Adam(lr=1e-3), MSELoss. ε: 1,0 → 0,02 линейно за 10 000 шагов. Условие решения: средняя награда по 100 последним эпизодам ≥ 195. Финальная ячейка сохраняет `results/05_dqn_training.csv`. Соответствует **таблице 6** отчёта.

**Источники:** Lapan M., глава 6; Mnih V. et al., 2013, 2015 (Nature).

---

## Структура репозитория

```
rl-lab/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── 01_openai_gym.ipynb
├── 02_cross_entropy.ipynb
├── 03_value_iteration.ipynb
├── 04_q_learning.ipynb
├── 05_dqn.ipynb
└── results/
    ├── 01_random_baseline.csv    (создаётся ноутбуком 01)
    ├── 02_cem_training.csv       (создаётся ноутбуком 02)
    ├── 03_value_iteration.csv    (создаётся ноутбуком 03)
    ├── 04_q_learning.csv         (создаётся ноутбуком 04)
    └── 05_dqn_training.csv       (создаётся ноутбуком 05)
```

## Запуск

Локально:

```bash
pip install -r requirements.txt
jupyter lab
```

В Colab: кнопки «Open in Colab» выше. Для ноутбука 05 (DQN) желательно включить GPU через **Runtime → Change runtime type → T4 GPU**, остальные работают на CPU за минуты.

## Литература

1. Lapan M. *Deep Reinforcement Learning Hands-On.* <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On>
2. Sutton R. S., Barto A. G. *Reinforcement Learning: An Introduction.* 2nd ed. Cambridge: The MIT Press, 2018. 552 p.
3. *Gymnasium documentation.* <https://gymnasium.farama.org/>
4. Mnih V. et al. *Playing Atari with Deep Reinforcement Learning.* 2013. <https://arxiv.org/abs/1312.5602>
5. Mnih V. et al. *Human-level control through deep reinforcement learning.* Nature, 2015. <https://www.nature.com/articles/nature14236>
6. PyTorch Documentation. <https://pytorch.org/docs/stable/index.html>
