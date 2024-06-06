import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import ssl
import re


# Загрузка моделі для українскої мови
nlp = spacy.load("uk_core_news_sm")

# Приклад текста
text = """Гоголь досі залишається нерозгаданою загадкою. Його переслідувала містика, і після його смерті залишилося більше питань, ніж відповідей. Вони дозволяють поглянути на творчість улюбленого письменника з абсолютно іншого боку, спробувати пояснити якісь протиріччя, невідповідності та побачити його не ідолом, а простою, неймовірно тонкою і талановитою людиною.
Гоголь відчував пристрасть до рукоділля. В’язав на спицях шарфи, кроїв сестрам сукні, ткав пояси, до літа шив собі шийні хустки.
Письменник обожнював мініатюрні видання. Не люблячи і не знаючи математики, він виписав математичну енциклопедію тільки тому, що вона була видана у форматі 10,5 × 7,5 см.
Гоголь любив готувати і пригощати друзів варениками та галушками. Один з улюблених його напоїв – козяче молоко, яке він варив особливим способом, додаючи ром. Це куховарство він називав гоголем-моголем та часто, сміючись, говорив: «Гоголь любить гоголь-моголь!»
Письменник ходив по вулицях і алеях зазвичай з лівого боку, тому постійно стикався з перехожими.
Гоголь дуже боявся грози. За словами сучасників, негода погано діяла на його слабкі нерви.
Він був вкрай сором’язливий. Як тільки у компанії з’являвся незнайомець, Гоголь зникав з кімнати.
Гоголь часто, коли писав, катав кульки з білого хліба. Друзям він казав, що це допомагає йому у вирішенні найскладніших завдань.
У кишенях у Гоголя завжди лежали солодощі. Живучи у готелі, він ніколи не дозволяв прислузі відносити поданий до чаю цукор, збирав його, ховав, а потім гриз шматки за роботою або розмовою.
Микола Васильович пристрасно захоплювався всім, що потрапляло в поле його зору. Історія рідної України була для нього одним з улюблених досліджень і захоплень. Саме ці дослідження підштовхнули його до написання епічної повісті «Тарас Бульба». Вона була вперше опублікована в збірці «Миргород» в 1835 році, один примірник  журналу Гоголь особисто вручив в руки панові Уварову – міністру народної освіти, для того, щоб той підніс його імператору Миколі I.
Сучасні фахівці в галузі психіатрії проаналізували тисячі документів і прийшли до абсолютно певного висновку про те, що ніякого психічного розладу у Гоголя не було. Можливо, він страждав на депресію, і якби до нього було застосовано правильне лікування, великий письменник прожив би набагато довше… """

# Обробка тексту за допомогою spaCy
doc = nlp(text)

# Токенизація
tokens = [token.text for token in doc]
print("Токени:", tokens)

# Видалення стоп-слів, пунктуації, чисел, и специальных символов
filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.like_num and token.text.isalpha()]
print("Відфільтровані токени:", filtered_tokens)

# Стемінг і лематизація
stems = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num and token.text.isalpha()]
print("Стеми:", stems)


# ENG TEXT



# Виключення перевірки SSL-сертификатов
ssl._create_default_https_context = ssl._create_unverified_context

# Загрузка данних
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Приклад текста
textEng = """Gogol is still an unsolved mystery. He was haunted by mysticism, and after his death there were more questions than answers. They allow us to look at the work of our favorite writer from a completely different perspective, to try to explain some contradictions and inconsistencies, and to see him not as an idol, but as a simple, incredibly subtle and talented person.
Gogol had a passion for needlework. He crocheted scarves, cut dresses for his sisters, wove belts, and sewed neck scarves for himself for the summer.
The writer adored miniature editions. Not liking and not knowing mathematics, he subscribed to a mathematical encyclopedia only because it was published in a 10.5 × 7.5 cm format.
Gogol loved to cook and treat his friends to dumplings and dumplings. One of his favorite drinks was goat's milk, which he brewed in a special way, adding rum. He called this cooking gogol-mogol and often said, laughing: “Gogol loves gogol-mogol!”
The writer usually walked along the streets and alleys on the left side, so he constantly encountered passers-by.
Gogol was very afraid of thunderstorms. According to his contemporaries, bad weather was bad for his weak nerves.
He was extremely shy. As soon as a stranger appeared in the company, Gogol would disappear from the room.
When writing, Gogol often rolled balls of white bread. He told his friends that it helped him solve the most difficult problems.
Gogol always had sweets in his pockets. While living in a hotel, he never allowed the servants to take away the sugar served for tea, collected it, hid it, and then gnawed on pieces with"""

# Токенізація
tokensEng = word_tokenize(textEng)
print("Tokens:", tokensEng)

# Видалення стоп-слов і знаків пунктуації
stop_wordsENG = set(stopwords.words('english'))
filtered_tokensENG = [word for word in tokensEng if word.lower() not in stop_wordsENG and word.isalnum()]
print("Filtered Tokens:", filtered_tokensENG)

# Стеммінг
stemmerENG = PorterStemmer()
stemsENG = [stemmerENG.stem(word) for word in filtered_tokensENG]
print("Stems:", stemsENG)

# Лемматизація
lemmatizer = WordNetLemmatizer()
lemmasENG = [lemmatizer.lemmatize(word) for word in filtered_tokensENG]
print("Lemmas:", lemmasENG)