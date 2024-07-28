from bytepiece import Trainer

import json
import logging
class corpus:
    def __iter__(self):
        f = 'pre_train/data/part.jsonl'
        with open(f) as f:
            for l in f:
                yield json.loads(l)['content']  # 每次返回一个Unicode

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('Starting training...')
    trainer = Trainer(order=6, max_vocab_size=80000, min_count=32, isolate_digits=True)
    try:
        trainer.train(corpus(), workers=1, batch_size=10)
        trainer.save('bytepiece.model')
        logging.debug('Training completed successfully.')
    except Exception as e:
        logging.error(f'Error during training: {e}')