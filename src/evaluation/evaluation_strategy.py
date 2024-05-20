class EvaluationStrategy:
    def validate(self, model, data_loader, device, loss_fn) -> dict:
        raise NotImplementedError
    
    def test(self, model, data_loader,  device, loss_fn) -> dict:
        raise NotImplementedError
    
    def custom_eval(self, model, data_loader, device, loss_fn, prefix) -> dict:
        raise NotImplementedError
    
    def unlearn_eval(self, model, test_data_loader, forgotten_dataloader, remembered_data_loader, device, loss_fn) -> dict:
        raise NotImplementedError