using TextAnalysis, Languages

# Importing own document chapters separately
pathname1 = "/Users/nfsturm/Documents/LTCF1.txt"
ltcf1 = FileDocument(pathname1)

pathname2 = "/Users/nfsturm/Documents/LTCF2.txt"
ltcf2 = FileDocument(pathname2)

pathname3 = "/Users/nfsturm/Documents/LTCF1.txt"
ltcf3 = FileDocument(pathname3)

pathname4 = "/Users/nfsturm/Documents/LTCF1.txt"
ltcf4 = FileDocument(pathname4)

ltcf1 = convert(StringDocument, ltcf1)
ltcf2 = convert(StringDocument, ltcf2)
ltcf3 = convert(StringDocument, ltcf3)
ltcf4 = convert(StringDocument, ltcf4)

# Create Corpus and do preprocessing
crps = Corpus([ltcf1, ltcf2, ltcf3, ltcf4])
prepare!(crps, strip_punctuation)
prepare!(crps, strip_articles)
prepare!(crps, strip_html_tags)
prepare!(crps, strip_numbers)
prepare!(crps, strip_prepositions)
prepare!(crps, strip_pronouns)
remove_words!(crps, ["was", "is", "were", "and", "or", "of"])


# Create lexicon with word counts
lex = lexicon(crps) # The next two lines must be executed separately
update_lexicon!(crps)

# Wordcount Dataframe
using DataFrames

wordcounts = DataFrame(Any[collect(keys(lex)), collect(values(lex))])
new_names = ["word", "count"]
rename!(wordcounts, Symbol.(new_names))

# Import Scored Sentiment Lexicon

using DelimitedFiles

afinn = readdlm("/Users/nfsturm/Documents/AFINN-en-165.txt", '\t')
afinn_df = convert(DataFrame, afinn)

afinn_names = ["word", "sentiment"]
rename!(afinn_df, Symbol.(afinn_names))
df = join(wordcounts, afinn_df, on = :word, kind = :inner)

# Import Categorical Sentiment Lexicon

using DelimitedFiles

bing_neg = readdlm("/Users/nfsturm/Documents/opinion-lexicon-English/negative-words.txt")
bing_pos = readdlm("/Users/nfsturm/Documents/opinion-lexicon-English/positive-words.txt")

bing_neg_df = convert(DataFrame, bing_neg)
bing_neg_name = ["word"]
rename!(bing_neg_df, Symbol.(bing_neg_name))

bing_pos_df = convert(DataFrame, bing_pos)
bing_pos_name = ["word"]
rename!(bing_pos_df, Symbol.(bing_pos_name))

# Explore sentiment data by category

df_bing_pos = join(wordcounts, bing_pos_df, on = :word, kind = :inner)
df_bing_neg = join(wordcounts, bing_neg_df, on = :word, kind = :inner)

df_bing_neg_10 = first(sort(df_bing_neg, [:count], rev = true), 10)
df_bing_pos_10 = first(sort(df_bing_pos, [:count], rev = true), 10)

# Plotting most frequent terms by sentiment

using Gadfly

plot(df_bing_pos_10, x = :word, y = :count, Geom.bar, Theme(bar_spacing=1mm, default_color = "#a7e0f4"))
plot(df_bing_neg_10, x = :word, y = :count, Geom.bar, Theme(bar_spacing=1mm, default_color = "#fdb9c9"))
