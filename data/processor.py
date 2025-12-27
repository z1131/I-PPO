"""
数据预处理器: 统一提取 QA 对，处理异常样本
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from configs.config import Config

logger = logging.getLogger("I-PPO")


class QAProcessor:
    """
    QA 处理器: 从不同格式的数据集中提取统一的 question/answer
    
    异常处理策略:
    - 失败样本跳过并记录日志
    - 不使用 mock 数据或降级处理
    """
    
    def __init__(self):
        self.failed_samples = []
        self.max_failures_log = getattr(Config, 'MAX_SAMPLE_FAILURES_LOG', 100)
        
    def process_dataset(self, ds_name: str, raw_data: Any) -> List[Dict[str, str]]:
        """
        处理整个数据集
        
        Args:
            ds_name: 数据集名称
            raw_data: 原始数据 (HuggingFace Dataset 或 ModelScope MsDataset)
            
        Returns:
            处理后的 QA 列表
        """
        results = []
        failed_count = 0
        
        for idx, item in enumerate(raw_data):
            try:
                q, a = self._extract_qa(ds_name, item)
                if q and a:
                    results.append({
                        'dataset': ds_name,
                        'question': str(q),
                        'answer': str(a)
                    })
                else:
                    failed_count += 1
                    self._log_failure(ds_name, idx, "Empty question or answer")
            except Exception as e:
                failed_count += 1
                self._log_failure(ds_name, idx, str(e))
                continue  # 跳过失败样本，继续处理
        
        if failed_count > 0:
            logger.warning(f"[{ds_name}] 处理完成，失败 {failed_count} 条样本")
        
        return results
    
    def _extract_qa(self, ds_name: str, item: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        根据数据集格式提取 question 和 answer
        
        Args:
            ds_name: 数据集名称
            item: 单条数据
            
        Returns:
            (question, answer) 元组
        """
        ds_name_lower = ds_name.lower()
        
        # LongBench 格式
        if 'longbench' in ds_name_lower:
            return self._extract_longbench(item)
        
        # QuALITY 格式
        if 'quality' in ds_name_lower:
            return self._extract_quality(item)
        
        # MATH 格式
        if 'math' in ds_name_lower:
            return self._extract_math(item)
        
        # ARC 格式
        if 'arc' in ds_name_lower:
            return self._extract_arc(item)
        
        # 通用 QA 格式 (CommonsenseQA, OpenBookQA 等)
        return self._extract_generic(item)
    
    def _extract_longbench(self, item: Dict) -> Tuple[str, str]:
        """LongBench 数据集格式"""
        context = item.get('context', '')
        question = item.get('input', '')
        
        # 拼接上下文和问题
        if context:
            full_question = f"{context}\n\nQuestion: {question}"
        else:
            full_question = question
        
        answers = item.get('answers', [])
        answer = answers[0] if answers else ''
        
        return full_question, answer
    
    def _extract_quality(self, item: Dict) -> Tuple[Optional[str], Optional[str]]:
        """QuALITY 数据集格式"""
        article = item.get('article', '')
        questions = item.get('questions', [])
        
        if not questions:
            return None, None
        
        # 取第一个问题
        q_data = questions[0] if isinstance(questions, list) else questions
        
        if isinstance(q_data, dict):
            question_text = q_data.get('question', '')
            options = q_data.get('options', [])
            gold_label = q_data.get('gold_label', 0)
            
            if options and isinstance(gold_label, int) and gold_label < len(options):
                answer = options[gold_label]
            else:
                answer = str(gold_label)
        else:
            question_text = str(q_data)
            answer = ''
        
        full_question = f"{article}\n\nQuestion: {question_text}"
        return full_question, answer
    
    def _extract_math(self, item: Dict) -> Tuple[str, str]:
        """MATH 数据集格式"""
        problem = item.get('problem', item.get('question', ''))
        solution = item.get('solution', item.get('answer', ''))
        return problem, solution
    
    def _extract_arc(self, item: Dict) -> Tuple[str, str]:
        """ARC 数据集格式"""
        question = item.get('question', '')
        answer_key = item.get('answerKey', '')
        
        # ARC 包含选项，可以拼接
        choices = item.get('choices', {})
        if choices:
            labels = choices.get('label', [])
            texts = choices.get('text', [])
            options_str = '\n'.join([f"{l}. {t}" for l, t in zip(labels, texts)])
            if options_str:
                question = f"{question}\n\nOptions:\n{options_str}"
        
        return question, answer_key
    
    def _extract_generic(self, item: Dict) -> Tuple[str, str]:
        """通用 QA 格式"""
        # 尝试多种常见字段名
        q = (item.get('question') or 
             item.get('problem') or 
             item.get('question_stem') or 
             item.get('input') or '')
        
        a = (item.get('answerKey') or 
             item.get('answer') or 
             item.get('solution') or 
             item.get('output') or '')
        
        # 如果 answer 是列表，取第一个
        if isinstance(a, list):
            a = a[0] if a else ''
        
        return str(q), str(a)
    
    def _log_failure(self, ds_name: str, idx: int, reason: str):
        """记录失败样本"""
        if len(self.failed_samples) < self.max_failures_log:
            self.failed_samples.append({
                'dataset': ds_name,
                'index': idx,
                'reason': reason
            })
            logger.debug(f"样本处理失败 [{ds_name}#{idx}]: {reason}")
    
    def get_failure_report(self) -> Dict[str, Any]:
        """获取失败报告"""
        return {
            'total_failures': len(self.failed_samples),
            'samples': self.failed_samples[:20],  # 只返回前 20 条
            'truncated': len(self.failed_samples) > 20
        }
