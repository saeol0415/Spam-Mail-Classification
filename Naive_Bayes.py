import string
import json

# Load training emails from JSON file
with open('Naive Bayes\Training_Emails.json', 'r') as file:
    email_data = json.load(file)

spam_emails = email_data['spam_emails']
ham_emails = email_data['ham_emails']

# 테스트 이메일
test_email = input() # Sample: 'buy now and save big'

# 이메일 전처리 함수
def preprocess(email):
    email = email.lower().translate(str.maketrans('', '', string.punctuation))
    return email.split()

# 단어 빈도 계산 함수
def calculate_word_freq(emails):
    word_freq = {}
    total_words = 0
    for email in emails:
        words = preprocess(email)
        total_words += len(words)
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return word_freq, total_words

# 단어 확률 계산 함수: P(w|Class)값 배열 (Laplace Smoothing)
def calculate_word_probs_from_freq(word_freq, total_words, vocab_size):
    word_probs = {}
    for word, freq in word_freq.items():
        word_probs[word] = (freq + 1) / (total_words + vocab_size)
    return word_probs

# 훈련 데이터에서 단어 확률 계산
spam_word_freq, spam_total_words = calculate_word_freq(spam_emails)
ham_word_freq, ham_total_words = calculate_word_freq(ham_emails)

all_words = set(spam_word_freq.keys()).union(set(ham_word_freq.keys()))
vocab_size = len(all_words)  # 고유 단어 수 (V)

spam_word_probs = calculate_word_probs_from_freq(spam_word_freq, spam_total_words, vocab_size)
ham_word_probs = calculate_word_probs_from_freq(ham_word_freq, ham_total_words, vocab_size)

# 스팸 및 햄 이메일의 사전 확률: P(Class)
P_spam = len(spam_emails) / (len(spam_emails) + len(ham_emails))
P_ham = len(ham_emails) / (len(spam_emails) + len(ham_emails))

# 이메일의 단어 확률 계산 함수: P(Class \cap Email)
def calculate_email_prob(email, word_probs, total_words, P_class):
    words = preprocess(email)
    prob = 1.0
    for word in words:
        if word in word_probs:
            prob *= word_probs[word]
        else:
            prob *= 1 / (total_words + vocab_size)
    return prob * P_class

# 조건부 확률 계산
P_email_given_spam = calculate_email_prob(test_email, spam_word_probs, spam_total_words, P_spam)
P_email_given_ham = calculate_email_prob(test_email, ham_word_probs, ham_total_words, P_ham)

# 전체 이메일 확률
P_email = P_email_given_spam + P_email_given_ham

# 스팸일 확률 계산: P(Spam|Email)
P_spam_given_email = P_email_given_spam / P_email

print(f'Probability that the email is spam: {P_spam_given_email:.4f}')