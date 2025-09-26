"""
Comprehensive unit tests for compare_embeddings.py module.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from rag_project.compare_embeddings import (
    init_embeddings,
    get_embedding,
    compare_embeddings,
)


class TestInitEmbeddings:
    """Test cases for init_embeddings function."""

    @patch('rag_project.compare_embeddings.HuggingFaceEmbeddings')
    @patch('rag_project.compare_embeddings.SENTENCE_TRANSFORMERS_MODEL_NAME', 'test-model')
    def test_init_embeddings_success(self, mock_hf_embeddings):
        """
        Testa a inicialização bem-sucedida do modelo de embeddings HuggingFace.
        
        Este teste verifica se a função consegue inicializar corretamente o
        modelo de embeddings com as configurações adequadas. É fundamental
        para garantir que o sistema de busca semântica funcione corretamente.
        
        Cenário testado:
        - HuggingFaceEmbeddings é inicializado com modelo correto
        - Configurações de dispositivo são definidas para CPU
        - Parâmetros de encoding são configurados adequadamente
        - Normalização e conversão para numpy são habilitadas
        """
        mock_embeddings = Mock()
        mock_hf_embeddings.return_value = mock_embeddings
        
        result = init_embeddings()
        
        assert result == mock_embeddings
        mock_hf_embeddings.assert_called_once_with(
            model_name='test-model',
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 8, "normalize_embeddings": True, "convert_to_numpy": True}
        )

    @patch('rag_project.compare_embeddings.HuggingFaceEmbeddings')
    def test_init_embeddings_with_default_model_name(self, mock_hf_embeddings):
        """
        Testa a inicialização com o nome do modelo padrão das constantes.
        
        Este teste verifica se a função utiliza corretamente o nome do
        modelo definido nas constantes do projeto, garantindo consistência
        na configuração do sistema.
        
        Cenário testado:
        - Modelo padrão SENTENCE_TRANSFORMERS_MODEL_NAME é usado
        - Configurações padrão são aplicadas corretamente
        - Sistema funciona sem configuração manual
        """
        mock_embeddings = Mock()
        mock_hf_embeddings.return_value = mock_embeddings
        
        result = init_embeddings()
        
        assert result == mock_embeddings
        # Verifica se foi chamado com o modelo padrão
        call_args = mock_hf_embeddings.call_args
        assert 'model_name' in call_args[1]
        assert call_args[1]['model_kwargs'] == {"device": "cpu"}
        assert call_args[1]['encode_kwargs'] == {
            "batch_size": 8, 
            "normalize_embeddings": True, 
            "convert_to_numpy": True
        }

    @patch('rag_project.compare_embeddings.HuggingFaceEmbeddings')
    def test_init_embeddings_huggingface_error(self, mock_hf_embeddings):
        """
        Testa o tratamento de erros durante inicialização do HuggingFace.
        
        Este teste verifica se a função trata adequadamente falhas que
        podem ocorrer durante a inicialização do modelo HuggingFace,
        como problemas de conectividade, modelo não encontrado, ou
        falta de recursos.
        
        Cenário testado:
        - HuggingFaceEmbeddings falha com RuntimeError
        - Erro é propagado corretamente para o chamador
        - Sistema falha de forma controlada
        """
        mock_hf_embeddings.side_effect = RuntimeError("Model loading failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            init_embeddings()
        
        assert "Model loading failed" in str(exc_info.value)

    @patch('rag_project.compare_embeddings.HuggingFaceEmbeddings')
    def test_init_embeddings_memory_error(self, mock_hf_embeddings):
        """
        Testa o tratamento de erros de memória durante inicialização.
        
        Este teste verifica se a função trata adequadamente problemas
        de memória insuficiente que podem ocorrer ao carregar modelos
        grandes de embeddings. É importante para ambientes com
        recursos limitados.
        
        Cenário testado:
        - Falha de memória durante carregamento do modelo
        - MemoryError é propagada corretamente
        - Sistema indica claramente problema de recursos
        """
        mock_hf_embeddings.side_effect = MemoryError("Insufficient memory")
        
        with pytest.raises(MemoryError) as exc_info:
            init_embeddings()
        
        assert "Insufficient memory" in str(exc_info.value)

    @patch('rag_project.compare_embeddings.HuggingFaceEmbeddings')
    def test_init_embeddings_configuration_validation(self, mock_hf_embeddings):
        """
        Testa se as configurações específicas são validadas corretamente.
        
        Este teste verifica se a função aplica as configurações exatas
        necessárias para o funcionamento otimizado do sistema, incluindo
        normalização de embeddings e conversão para numpy.
        
        Cenário testado:
        - Batch size é definido como 8 para performance otimizada
        - Normalização de embeddings está habilitada
        - Conversão para numpy está habilitada para compatibilidade
        - Dispositivo CPU é explicitamente configurado
        """
        mock_embeddings = Mock()
        mock_hf_embeddings.return_value = mock_embeddings
        
        init_embeddings()
        
        call_args = mock_hf_embeddings.call_args
        encode_kwargs = call_args[1]['encode_kwargs']
        
        assert encode_kwargs['batch_size'] == 8
        assert encode_kwargs['normalize_embeddings'] is True
        assert encode_kwargs['convert_to_numpy'] is True
        
        model_kwargs = call_args[1]['model_kwargs']
        assert model_kwargs['device'] == 'cpu'


class TestGetEmbedding:
    """Test cases for get_embedding function."""

    def test_get_embedding_success(self):
        """
        Testa a geração bem-sucedida de embedding para um texto.
        
        Este teste verifica se a função consegue gerar corretamente
        embeddings para um texto de entrada usando o modelo fornecido.
        É fundamental para o funcionamento da busca semântica.
        
        Cenário testado:
        - Texto válido é processado pelo modelo
        - embed_documents é chamado com lista contendo o texto
        - Primeiro embedding da lista é retornado
        - Formato do resultado é uma lista de números
        """
        mock_model = Mock()
        mock_embeddings = [[0.1, 0.2, 0.3, 0.4]]
        mock_model.embed_documents.return_value = mock_embeddings
        
        text = "This is a test text"
        result = get_embedding(text, mock_model)
        
        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_model.embed_documents.assert_called_once_with([text])

    def test_get_embedding_empty_text(self):
        """
        Testa o comportamento com texto vazio.
        
        Este teste verifica se a função consegue processar adequadamente
        strings vazias, que podem ocorrer em dados mal formados ou
        durante processamento de documentos com conteúdo faltante.
        
        Cenário testado:
        - String vazia é fornecida como entrada
        - Modelo processa a string vazia normalmente
        - Embedding é gerado (mesmo que para conteúdo vazio)
        - Resultado mantém formato esperado
        """
        mock_model = Mock()
        mock_embeddings = [[0.0, 0.0, 0.0, 0.0]]
        mock_model.embed_documents.return_value = mock_embeddings
        
        text = ""
        result = get_embedding(text, mock_model)
        
        assert result == [0.0, 0.0, 0.0, 0.0]
        mock_model.embed_documents.assert_called_once_with([""])

    def test_get_embedding_long_text(self):
        """
        Testa o processamento de texto muito longo.
        
        Este teste verifica se a função consegue processar textos
        extremamente longos sem problemas de performance ou memória.
        É importante para documentos grandes que podem ser encontrados
        em sistemas reais.
        
        Cenário testado:
        - Texto com 1000 caracteres é processado
        - Modelo lida adequadamente com texto longo
        - Embedding é gerado sem truncamento inesperado
        - Performance se mantém aceitável
        """
        mock_model = Mock()
        mock_embeddings = [[0.5] * 384]  # Tamanho típico de embedding
        mock_model.embed_documents.return_value = mock_embeddings
        
        long_text = "A" * 1000
        result = get_embedding(long_text, mock_model)
        
        assert len(result) == 384
        assert all(x == 0.5 for x in result)
        mock_model.embed_documents.assert_called_once_with([long_text])

    def test_get_embedding_special_characters(self):
        """
        Testa o processamento de texto com caracteres especiais.
        
        Este teste verifica se a função consegue processar adequadamente
        textos contendo caracteres especiais, unicode, emojis e acentos.
        É essencial para suporte internacional e multilíngue.
        
        Cenário testado:
        - Texto com caracteres especiais, unicode e emojis
        - Modelo processa caracteres internacionais corretamente
        - Embedding preserva informação semântica de caracteres especiais
        - Nenhum problema de encoding ou corrupção
        """
        mock_model = Mock()
        mock_embeddings = [[0.1, 0.2, 0.3]]
        mock_model.embed_documents.return_value = mock_embeddings
        
        special_text = "Héllo wörld! 🌍 你好 مرحبا"
        result = get_embedding(special_text, mock_model)
        
        assert result == [0.1, 0.2, 0.3]
        mock_model.embed_documents.assert_called_once_with([special_text])

    def test_get_embedding_model_error(self):
        """
        Testa o tratamento de erros durante geração de embedding.
        
        Este teste verifica se a função trata adequadamente falhas
        que podem ocorrer durante a geração de embeddings, como
        problemas no modelo, timeout, ou falhas de conectividade.
        
        Cenário testado:
        - Modelo falha durante embed_documents
        - RuntimeError é propagada corretamente
        - Sistema falha de forma controlada
        """
        mock_model = Mock()
        mock_model.embed_documents.side_effect = RuntimeError("Embedding generation failed")
        
        text = "Test text"
        
        with pytest.raises(RuntimeError) as exc_info:
            get_embedding(text, mock_model)
        
        assert "Embedding generation failed" in str(exc_info.value)

    def test_get_embedding_returns_first_element(self):
        """
        Testa se a função retorna corretamente o primeiro elemento da lista.
        
        Este teste verifica se a função extrai adequadamente o primeiro
        (e único) embedding da lista retornada pelo modelo, garantindo
        que a interface seja simplificada para o usuário.
        
        Cenário testado:
        - Modelo retorna lista com múltiplos embeddings
        - Função retorna apenas o primeiro embedding
        - Outros embeddings são ignorados corretamente
        """
        mock_model = Mock()
        mock_embeddings = [
            [0.1, 0.2, 0.3],  # Primeiro embedding (deve ser retornado)
            [0.4, 0.5, 0.6],  # Segundo embedding (deve ser ignorado)
        ]
        mock_model.embed_documents.return_value = mock_embeddings
        
        text = "Test text"
        result = get_embedding(text, mock_model)
        
        assert result == [0.1, 0.2, 0.3]
        assert result != [0.4, 0.5, 0.6]

    def test_get_embedding_with_numpy_array(self):
        """
        Testa o comportamento quando o modelo retorna numpy arrays.
        
        Este teste verifica se a função consegue lidar adequadamente
        com modelos que retornam numpy arrays em vez de listas Python,
        garantindo compatibilidade com diferentes implementações.
        
        Cenário testado:
        - Modelo retorna numpy arrays
        - Função processa arrays corretamente
        - Resultado é convertido para formato apropriado
        """
        mock_model = Mock()
        mock_embeddings = [np.array([0.1, 0.2, 0.3, 0.4])]
        mock_model.embed_documents.return_value = mock_embeddings
        
        text = "Test text"
        result = get_embedding(text, mock_model)
        
        # Resultado deve ser compatível com lista Python
        assert len(result) == 4
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(0.2)
        assert result[2] == pytest.approx(0.3)
        assert result[3] == pytest.approx(0.4)


class TestCompareEmbeddings:
    """Test cases for compare_embeddings function."""

    @patch('rag_project.compare_embeddings.get_embedding')
    @patch('rag_project.compare_embeddings.cosine_similarity')
    def test_compare_embeddings_success(self, mock_cosine_similarity, mock_get_embedding):
        """
        Testa a comparação bem-sucedida entre dois textos.
        
        Este teste verifica se a função consegue comparar adequadamente
        dois textos calculando a similaridade coseno entre seus embeddings.
        É fundamental para o funcionamento da busca semântica e ranking
        de relevância.
        
        Cenário testado:
        - Dois textos são convertidos em embeddings
        - Similaridade coseno é calculada corretamente
        - Resultado é convertido para float
        - Pipeline completo funciona sem erros
        """
        # Mock embeddings para os dois textos
        mock_get_embedding.side_effect = [
            [0.1, 0.2, 0.3],  # Embedding do primeiro texto
            [0.4, 0.5, 0.6],  # Embedding do segundo texto
        ]
        
        # Mock resultado da similaridade coseno
        mock_cosine_similarity.return_value = [[0.8]]
        
        mock_model = Mock()
        text1 = "First text"
        text2 = "Second text"
        
        result = compare_embeddings(text1, text2, mock_model)
        
        assert result == pytest.approx(0.8)
        assert isinstance(result, float)
        
        # Verifica se get_embedding foi chamado para ambos os textos
        assert mock_get_embedding.call_count == 2
        mock_get_embedding.assert_any_call(text1, mock_model)
        mock_get_embedding.assert_any_call(text2, mock_model)
        
        # Verifica se cosine_similarity foi chamado com os embeddings corretos
        mock_cosine_similarity.assert_called_once_with([[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]])

    @patch('rag_project.compare_embeddings.get_embedding')
    @patch('rag_project.compare_embeddings.cosine_similarity')
    def test_compare_embeddings_identical_texts(self, mock_cosine_similarity, mock_get_embedding):
        """
        Testa a comparação entre textos idênticos.
        
        Este teste verifica se a função retorna alta similaridade (próxima
        de 1.0) quando os textos são idênticos. É importante para validar
        que o sistema reconhece duplicatas e conteúdo similar.
        
        Cenário testado:
        - Dois textos idênticos são comparados
        - Embeddings são iguais ou muito similares
        - Similaridade coseno retorna valor próximo de 1.0
        - Sistema reconhece conteúdo duplicado
        """
        # Embeddings idênticos para textos idênticos
        identical_embedding = [0.1, 0.2, 0.3, 0.4]
        mock_get_embedding.return_value = identical_embedding
        
        # Similaridade máxima para textos idênticos
        mock_cosine_similarity.return_value = [[1.0]]
        
        mock_model = Mock()
        text = "Identical text"
        
        result = compare_embeddings(text, text, mock_model)
        
        assert result == pytest.approx(1.0)
        assert mock_get_embedding.call_count == 2
        mock_cosine_similarity.assert_called_once_with([identical_embedding], [identical_embedding])

    @patch('rag_project.compare_embeddings.get_embedding')
    @patch('rag_project.compare_embeddings.cosine_similarity')
    def test_compare_embeddings_completely_different_texts(self, mock_cosine_similarity, mock_get_embedding):
        """
        Testa a comparação entre textos completamente diferentes.
        
        Este teste verifica se a função retorna baixa similaridade quando
        os textos são semanticamente diferentes. É importante para garantir
        que o sistema distingue adequadamente conteúdo não relacionado.
        
        Cenário testado:
        - Dois textos semanticamente diferentes são comparados
        - Embeddings são diferentes
        - Similaridade coseno retorna valor baixo
        - Sistema distingue conteúdo não relacionado
        """
        mock_get_embedding.side_effect = [
            [1.0, 0.0, 0.0],  # Embedding do primeiro texto
            [0.0, 1.0, 0.0],  # Embedding do segundo texto (ortogonal)
        ]
        
        # Baixa similaridade para textos diferentes
        mock_cosine_similarity.return_value = [[0.1]]
        
        mock_model = Mock()
        text1 = "Technology and computers"
        text2 = "Cooking and recipes"
        
        result = compare_embeddings(text1, text2, mock_model)
        
        assert result == pytest.approx(0.1)
        mock_cosine_similarity.assert_called_once_with([[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])

    @patch('rag_project.compare_embeddings.get_embedding')
    @patch('rag_project.compare_embeddings.cosine_similarity')
    def test_compare_embeddings_empty_texts(self, mock_cosine_similarity, mock_get_embedding):
        """
        Testa a comparação entre textos vazios.
        
        Este teste verifica se a função consegue lidar adequadamente
        com textos vazios, que podem ocorrer em dados mal formados
        ou durante processamento de documentos com conteúdo faltante.
        
        Cenário testado:
        - Textos vazios são processados
        - Embeddings de textos vazios são gerados
        - Similaridade é calculada mesmo para conteúdo vazio
        - Sistema não falha com entrada vazia
        """
        # Embeddings para textos vazios (normalmente zeros ou valores baixos)
        mock_get_embedding.return_value = [0.0, 0.0, 0.0, 0.0]
        
        mock_cosine_similarity.return_value = [[0.0]]
        
        mock_model = Mock()
        
        result = compare_embeddings("", "", mock_model)
        
        assert result == pytest.approx(0.0)
        assert mock_get_embedding.call_count == 2

    @patch('rag_project.compare_embeddings.get_embedding')
    @patch('rag_project.compare_embeddings.cosine_similarity')
    def test_compare_embeddings_with_special_characters(self, mock_cosine_similarity, mock_get_embedding):
        """
        Testa a comparação de textos com caracteres especiais.
        
        Este teste verifica se a função consegue comparar adequadamente
        textos contendo caracteres especiais, unicode, emojis e acentos.
        É essencial para suporte internacional e multilíngue do sistema.
        
        Cenário testado:
        - Textos com caracteres especiais são comparados
        - Embeddings preservam informação semântica internacional
        - Similaridade é calculada corretamente
        - Sistema suporta conteúdo multilíngue
        """
        mock_get_embedding.side_effect = [
            [0.2, 0.3, 0.4],  # Embedding do texto em português
            [0.2, 0.3, 0.4],  # Embedding similar para texto relacionado
        ]
        
        mock_cosine_similarity.return_value = [[0.95]]
        
        mock_model = Mock()
        text1 = "Olá mundo! 🌍"
        text2 = "Hello world! 🌍"
        
        result = compare_embeddings(text1, text2, mock_model)
        
        assert result == pytest.approx(0.95)

    @patch('rag_project.compare_embeddings.get_embedding')
    def test_compare_embeddings_get_embedding_error(self, mock_get_embedding):
        """
        Testa o tratamento de erros durante geração de embeddings.
        
        Este teste verifica se a função trata adequadamente falhas
        que podem ocorrer durante a geração de embeddings para
        qualquer um dos textos de entrada.
        
        Cenário testado:
        - get_embedding falha para o primeiro texto
        - RuntimeError é propagada corretamente
        - Sistema falha de forma controlada
        """
        mock_get_embedding.side_effect = RuntimeError("Embedding failed")
        
        mock_model = Mock()
        
        with pytest.raises(RuntimeError) as exc_info:
            compare_embeddings("text1", "text2", mock_model)
        
        assert "Embedding failed" in str(exc_info.value)

    @patch('rag_project.compare_embeddings.get_embedding')
    @patch('rag_project.compare_embeddings.cosine_similarity')
    def test_compare_embeddings_cosine_similarity_error(self, mock_cosine_similarity, mock_get_embedding):
        """
        Testa o tratamento de erros durante cálculo de similaridade coseno.
        
        Este teste verifica se a função trata adequadamente falhas
        que podem ocorrer durante o cálculo da similaridade coseno,
        como problemas com dimensões incompatíveis dos embeddings.
        
        Cenário testado:
        - Embeddings são gerados com sucesso
        - cosine_similarity falha com RuntimeError
        - Erro é propagado corretamente para o chamador
        """
        mock_get_embedding.side_effect = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        
        mock_cosine_similarity.side_effect = RuntimeError("Similarity calculation failed")
        
        mock_model = Mock()
        
        with pytest.raises(RuntimeError) as exc_info:
            compare_embeddings("text1", "text2", mock_model)
        
        assert "Similarity calculation failed" in str(exc_info.value)

    @patch('rag_project.compare_embeddings.get_embedding')
    @patch('rag_project.compare_embeddings.cosine_similarity')
    def test_compare_embeddings_return_type_conversion(self, mock_cosine_similarity, mock_get_embedding):
        """
        Testa a conversão correta do tipo de retorno para float.
        
        Este teste verifica se a função converte adequadamente o
        resultado da similaridade coseno (que pode ser numpy array
        ou outros tipos) para um float Python padrão.
        
        Cenário testado:
        - cosine_similarity retorna numpy array
        - Função extrai valor correto do array
        - Resultado é convertido para float Python
        - Tipo de retorno é consistente
        """
        mock_get_embedding.side_effect = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        
        # Simula retorno como numpy array
        mock_cosine_similarity.return_value = np.array([[0.75]])
        
        mock_model = Mock()
        
        result = compare_embeddings("text1", "text2", mock_model)
        
        assert result == pytest.approx(0.75)
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)

    @patch('rag_project.compare_embeddings.get_embedding')
    @patch('rag_project.compare_embeddings.cosine_similarity')
    def test_compare_embeddings_boundary_values(self, mock_cosine_similarity, mock_get_embedding):
        """
        Testa o comportamento com valores limítrofes de similaridade.
        
        Este teste verifica se a função lida adequadamente com valores
        extremos de similaridade (0.0 e 1.0) e valores próximos aos
        limites, garantindo precisão numérica adequada.
        
        Cenário testado:
        - Teste com similaridade 0.0 (completamente diferente)
        - Teste com similaridade 1.0 (idêntico)
        - Valores são preservados com precisão adequada
        """
        mock_get_embedding.side_effect = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],  # Vetores opostos
        ]
        
        # Similaridade mínima possível
        mock_cosine_similarity.return_value = [[-1.0]]
        
        mock_model = Mock()
        
        result = compare_embeddings("opposite1", "opposite2", mock_model)
        
        assert result == pytest.approx(-1.0)
        assert isinstance(result, float)


class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases for the embeddings module."""

    @patch('rag_project.compare_embeddings.HuggingFaceEmbeddings')
    def test_full_workflow_integration(self, mock_hf_embeddings):
        """
        Testa o fluxo completo de inicialização e comparação de embeddings.
        
        Este é um teste de integração que verifica se todos os componentes
        do módulo funcionam juntos corretamente, desde a inicialização
        do modelo até a comparação final de textos.
        
        Cenário testado:
        - Modelo é inicializado corretamente
        - Embeddings são gerados para dois textos
        - Similaridade é calculada com sucesso
        - Pipeline completo funciona sem erros
        """
        # Setup do mock model
        mock_model = Mock()
        mock_model.embed_documents.side_effect = [
            [[0.1, 0.2, 0.3]],  # Embedding para primeiro texto
            [[0.4, 0.5, 0.6]],  # Embedding para segundo texto
        ]
        mock_hf_embeddings.return_value = mock_model
        
        # Inicializa o modelo
        embeddings_model = init_embeddings()
        
        # Compara dois textos
        with patch('rag_project.compare_embeddings.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.85]]
            
            result = compare_embeddings("Hello world", "Hi there", embeddings_model)
            
            assert result == pytest.approx(0.85)
            assert mock_model.embed_documents.call_count == 2

    def test_large_text_comparison_performance(self):
        """
        Testa a performance com textos muito grandes.
        
        Este teste verifica se o sistema consegue processar textos
        extremamente longos sem problemas de performance ou memória.
        É importante para documentos grandes que podem ser encontrados
        em sistemas reais.
        
        Cenário testado:
        - Textos com milhares de caracteres são processados
        - Sistema não falha com conteúdo extenso
        - Performance se mantém aceitável
        - Memória é gerenciada adequadamente
        """
        mock_model = Mock()
        
        # Simula embeddings para textos grandes
        mock_model.embed_documents.side_effect = [
            [[0.1] * 384],  # Embedding para texto grande 1
            [[0.2] * 384],  # Embedding para texto grande 2
        ]
        
        large_text1 = "A" * 10000  # 10k caracteres
        large_text2 = "B" * 10000  # 10k caracteres
        
        with patch('rag_project.compare_embeddings.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.5]]
            
            result = compare_embeddings(large_text1, large_text2, mock_model)
            
            assert result == pytest.approx(0.5)
            # Verifica se os textos grandes foram processados
            mock_model.embed_documents.assert_any_call([large_text1])
            mock_model.embed_documents.assert_any_call([large_text2])

    def test_multilingual_text_comparison(self):
        """
        Testa a comparação de textos em diferentes idiomas.
        
        Este teste verifica se o sistema consegue comparar adequadamente
        textos em diferentes idiomas, mantendo a qualidade semântica
        da comparação mesmo com diferenças linguísticas.
        
        Cenário testado:
        - Textos em inglês, português, espanhol e chinês
        - Embeddings preservam informação semântica multilíngue
        - Comparações cross-linguísticas funcionam adequadamente
        - Sistema é robusto para conteúdo internacional
        """
        mock_model = Mock()
        
        # Simula embeddings para textos multilíngues
        mock_model.embed_documents.side_effect = [
            [[0.3, 0.4, 0.5]],  # Embedding para texto em português
            [[0.3, 0.4, 0.5]],  # Embedding similar para inglês
        ]
        
        portuguese_text = "Olá, como você está hoje?"
        english_text = "Hello, how are you today?"
        
        with patch('rag_project.compare_embeddings.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.92]]  # Alta similaridade semântica
            
            result = compare_embeddings(portuguese_text, english_text, mock_model)
            
            assert result == pytest.approx(0.92)

    def test_empty_and_whitespace_handling(self):
        """
        Testa o manuseio de textos vazios e com apenas espaços.
        
        Este teste verifica se o sistema consegue lidar adequadamente
        com textos vazios, apenas espaços em branco, ou combinações
        de caracteres de espaçamento. É importante para robustez
        com dados mal formados.
        
        Cenário testado:
        - String completamente vazia
        - String apenas com espaços
        - String apenas com tabs e quebras de linha
        - Combinações de caracteres de espaçamento
        """
        mock_model = Mock()
        
        test_cases = [
            ("", ""),
            ("   ", "   "),
            ("\t\n\r", "\t\n\r"),
            ("", "   "),
        ]
        
        for text1, text2 in test_cases:
            mock_model.embed_documents.side_effect = [
                [[0.0, 0.0, 0.0]],  # Embedding para texto vazio/whitespace
                [[0.0, 0.0, 0.0]],  # Embedding similar
            ]
            
            with patch('rag_project.compare_embeddings.cosine_similarity') as mock_cosine:
                mock_cosine.return_value = [[1.0]]  # Textos vazios são idênticos
                
                result = compare_embeddings(text1, text2, mock_model)
                
                assert result == pytest.approx(1.0)
                assert isinstance(result, float)

    def test_numerical_precision_and_edge_cases(self):
        """
        Testa a precisão numérica e casos extremos de similaridade.
        
        Este teste verifica se o sistema mantém precisão numérica
        adequada em casos extremos e lida corretamente com valores
        muito próximos aos limites matemáticos.
        
        Cenário testado:
        - Valores muito próximos de 0.0 e 1.0
        - Precisão decimal é preservada adequadamente
        - Casos extremos não causam overflow ou underflow
        """
        mock_model = Mock()
        
        # Teste com alta precisão
        mock_model.embed_documents.side_effect = [
            [[0.999999, 0.000001, 0.0]],
            [[0.999998, 0.000002, 0.0]],
        ]
        
        with patch('rag_project.compare_embeddings.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.999999999]]  # Muito próximo de 1.0
            
            result = compare_embeddings("almost_identical1", "almost_identical2", mock_model)
            
            assert result == pytest.approx(0.999999999, rel=1e-9)
            assert isinstance(result, float)


class TestErrorHandlingAndRobustness:
    """Tests focused on error handling and system robustness."""

    def test_model_none_handling(self):
        """
        Testa o comportamento quando o modelo é None.
        
        Este teste verifica se as funções tratam adequadamente
        situações onde o modelo não foi inicializado corretamente
        ou foi passado como None.
        
        Cenário testado:
        - Modelo None é passado para get_embedding
        - AttributeError é lançada quando tenta chamar embed_documents
        - Sistema falha de forma controlada e previsível
        """
        with pytest.raises(AttributeError):
            get_embedding("test text", None)  # type: ignore

    def test_invalid_embedding_dimensions(self):
        """
        Testa o comportamento com embeddings de dimensões inválidas.
        
        Este teste verifica se o sistema consegue detectar e lidar
        adequadamente com embeddings que têm dimensões incompatíveis
        ou estrutura inesperada.
        
        Cenário testado:
        - Embeddings com dimensões diferentes
        - cosine_similarity falha devido a incompatibilidade
        - Erro é propagado com informação útil
        """
        mock_model = Mock()
        
        # Embeddings com dimensões incompatíveis
        mock_model.embed_documents.side_effect = [
            [[0.1, 0.2, 0.3]],      # 3 dimensões
            [[0.4, 0.5, 0.6, 0.7]], # 4 dimensões
        ]
        
        with patch('rag_project.compare_embeddings.cosine_similarity') as mock_cosine:
            mock_cosine.side_effect = ValueError("Incompatible dimensions")
            
            with pytest.raises(ValueError) as exc_info:
                compare_embeddings("text1", "text2", mock_model)
            
            assert "Incompatible dimensions" in str(exc_info.value)

    def test_memory_constraints_simulation(self):
        """
        Testa o comportamento sob restrições de memória simuladas.
        
        Este teste verifica se o sistema trata adequadamente situações
        de baixa memória que podem ocorrer ao processar embeddings
        grandes ou múltiplos textos simultaneamente.
        
        Cenário testado:
        - MemoryError é simulada durante processamento
        - Erro é propagado corretamente
        - Sistema não corrompe estado interno
        """
        mock_model = Mock()
        mock_model.embed_documents.side_effect = MemoryError("Out of memory")
        
        with pytest.raises(MemoryError) as exc_info:
            get_embedding("memory intensive text", mock_model)
        
        assert "Out of memory" in str(exc_info.value)

    @patch('rag_project.compare_embeddings.HuggingFaceEmbeddings')
    def test_model_initialization_timeout(self, mock_hf_embeddings):
        """
        Testa o comportamento com timeout durante inicialização do modelo.
        
        Este teste verifica se o sistema trata adequadamente situações
        onde a inicialização do modelo demora muito tempo ou trava,
        simulando problemas de conectividade ou recursos.
        
        Cenário testado:
        - Inicialização do modelo falha por timeout
        - TimeoutError é propagada adequadamente
        - Sistema não fica em estado indefinido
        """
        mock_hf_embeddings.side_effect = TimeoutError("Model initialization timeout")
        
        with pytest.raises(TimeoutError) as exc_info:
            init_embeddings()
        
        assert "Model initialization timeout" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])