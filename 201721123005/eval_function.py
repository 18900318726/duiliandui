import collections
import math
#是一个评测翻译的质量的算法，包含的就是该算法的程序，在网络训练过程中作为loss指标进行优化。
#实现bleu得分的计算，从输入段提取给定最大顺序的所有n-grams。
def _get_ngrams(segment, max_order):
  #segment:将从中提取n-grams的文本段，max_order:此返回的n-grams的令牌的最大长度
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1 #计数每个n-gram出现的次数
  return ngram_counts



#根据一个或多个引用计算翻译片段的BLEU分数。
def compute_bleu(reference_corpus, translation_corpus, max_order=4,smooth=False):
  #reference_corpus：每次翻译的参考文献列表。每个引用都应该被标记为标记列表。
  #translation_corpus：要评分的翻译列表。每个翻译都应该被标记成一个标记列表。
  # max_order:计算BLEU分数时使用的最大n-gram顺序。
  #smooth：是否应用平滑
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)
