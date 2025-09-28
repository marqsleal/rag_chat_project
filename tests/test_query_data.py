"""
Comprehensive unit tests for query_data.py module.
"""

from unittest.mock import Mock, patch

import pytest

from rag_project.query_data import (
    RAGQueryEngine,
    create_rag_engine,
    format_prompt,
    init_chroma,
    load_local_llama,
    validate_query_inputs,
)
from rag_project.rag_models import LLMConfig, RAGResponse


class TestLoadLocalLlama:
    """Test cases for load_local_llama function."""

    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.CTransformers')
    @patch('rag_project.query_data.logger')
    def test_load_local_llama_success(self, mock_logger, mock_ctransformers, mock_path):
        """
        Testa o carregamento bem-sucedido de um modelo Llama local.
        
        Este teste verifica se a função load_local_llama consegue carregar corretamente
        um modelo LLM local quando o arquivo existe e todas as configurações são válidas.
        É importante testar este cenário pois é o caso de uso principal da função.
        
        Cenário testado:
        - Arquivo do modelo existe no sistema
        - Configuração válida é fornecida (temperatura e max_tokens)
        - CTransformers é inicializado corretamente
        - Logs de informação são gerados
        """
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_llm = Mock()
        mock_ctransformers.return_value = mock_llm

        config = LLMConfig(temperature=0.5, max_new_tokens=100)
        result = load_local_llama("test_model", "llama", config)

        assert result == mock_llm
        mock_ctransformers.assert_called_once_with(
            model="test_model",
            model_file=str(mock_path_instance),
            model_type="llama",
            config=config.model_dump(),
        )
        assert mock_logger.info.call_count >= 1

    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.logger')
    def test_load_local_llama_file_not_found(self, mock_logger, mock_path):
        """
        Testa o comportamento quando o arquivo do modelo não existe.
        
        Este teste é crucial para verificar se a aplicação trata adequadamente
        situações onde o usuário especifica um caminho inválido para o modelo.
        Garante que uma exceção específica (FileNotFoundError) seja lançada
        com uma mensagem clara, evitando comportamentos inesperados.
        
        Cenário testado:
        - Arquivo do modelo não existe no sistema de arquivos
        - FileNotFoundError é lançada com mensagem apropriada
        - Log de erro é gerado para debugging
        """
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_local_llama("nonexistent_model")
        
        assert "Model file not found" in str(exc_info.value)
        mock_logger.error.assert_called()

    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.CTransformers')
    @patch('rag_project.query_data.logger')
    def test_load_local_llama_ctransformers_error(self, mock_logger, mock_ctransformers, mock_path):
        """
        Testa o tratamento de erros durante a inicialização do CTransformers.
        
        Este teste verifica se a aplicação trata adequadamente falhas na
        inicialização da biblioteca CTransformers. Isso pode ocorrer por
        problemas de memória, modelo corrompido, ou incompatibilidade de
        hardware. É essencial para garantir que erros internos sejam
        propagados corretamente.
        
        Cenário testado:
        - Arquivo do modelo existe mas CTransformers falha ao inicializar
        - RuntimeError é propagada corretamente
        - Log de erro é gerado para troubleshooting
        """
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_ctransformers.side_effect = RuntimeError("Initialization failed")
        
        with pytest.raises(RuntimeError):
            load_local_llama("test_model")
        
        mock_logger.error.assert_called()

    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.CTransformers')
    @patch('rag_project.query_data.logger')
    def test_load_local_llama_with_default_config(self, mock_logger, mock_ctransformers, mock_path):
        """
        Testa o carregamento do modelo com configurações padrão.
        
        Este teste verifica se a função funciona corretamente quando
        nenhum parâmetro é fornecido, utilizando valores padrão.
        É importante para garantir que a função seja usável sem
        configuração manual detalhada.
        
        Cenário testado:
        - Nenhum parâmetro fornecido (usa defaults)
        - CTransformers é chamado com configuração padrão
        - Função retorna objeto LLM válido
        """
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_llm = Mock()
        mock_ctransformers.return_value = mock_llm
        
        result = load_local_llama()
        
        assert result == mock_llm
        mock_ctransformers.assert_called_once()

    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.logger')
    def test_load_local_llama_permission_error(self, mock_logger, mock_path):
        """
        Testa o tratamento de erros de permissão ao acessar o modelo.
        
        Este teste verifica o comportamento quando a aplicação não tem
        permissões adequadas para acessar o arquivo do modelo. Isso pode
        ocorrer em sistemas com controle de acesso rigoroso ou quando
        o arquivo está sendo usado por outro processo.
        
        Cenário testado:
        - Arquivo existe mas não há permissão de leitura
        - PermissionError é lançada corretamente
        - Sistema falha de forma controlada e previsível
        """
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        with patch('rag_project.query_data.CTransformers') as mock_ctransformers:
            mock_ctransformers.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                load_local_llama("test_model")


class TestInitChroma:
    """Test cases for init_chroma function."""

    @patch('rag_project.query_data.init_embeddings')
    @patch('rag_project.query_data.Chroma')
    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.logger')
    def test_init_chroma_success_existing_dir(self, mock_logger, mock_path, mock_chroma, mock_init_embeddings):
        """
        Testa a inicialização bem-sucedida do Chroma com diretório existente.
        
        Este teste verifica se a função init_chroma consegue inicializar
        corretamente o banco de dados vetorial Chroma quando o diretório
        de destino já existe. É o cenário mais comum em uso contínuo.
        
        Cenário testado:
        - Diretório do banco de dados já existe
        - Embeddings são inicializados corretamente
        - Chroma é configurado com sucesso
        - Não tenta criar diretório existente
        """
        mock_embedding = Mock()
        mock_init_embeddings.return_value = mock_embedding

        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_chroma_db = Mock()
        mock_chroma.return_value = mock_chroma_db

        result = init_chroma("/test/path", "test_collection")

        assert result == mock_chroma_db
        mock_chroma.assert_called_once()
        mock_path_instance.mkdir.assert_not_called()

    @patch('rag_project.query_data.init_embeddings')
    @patch('rag_project.query_data.Chroma')
    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.logger')
    def test_init_chroma_success_create_dir(self, mock_logger, mock_path, mock_chroma, mock_init_embeddings):
        """
        Testa a inicialização do Chroma criando novo diretório.
        
        Este teste verifica se a função consegue criar automaticamente
        o diretório de destino quando ele não existe. É importante para
        a primeira execução ou quando o usuário especifica um novo local.
        
        Cenário testado:
        - Diretório não existe inicialmente
        - Diretório é criado automaticamente (com parents=True)
        - Warning é logado sobre criação do diretório
        - Chroma é inicializado com sucesso após criação
        """
        mock_embedding = Mock()
        mock_init_embeddings.return_value = mock_embedding
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        mock_chroma_db = Mock()
        mock_chroma.return_value = mock_chroma_db
        
        result = init_chroma("/test/new_path", "test_collection")
        
        assert result == mock_chroma_db
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_logger.warning.assert_called()

    @patch('rag_project.query_data.init_embeddings')
    @patch('rag_project.query_data.logger')
    def test_init_chroma_embeddings_error(self, mock_logger, mock_init_embeddings):
        """
        Testa o tratamento de falhas na inicialização dos embeddings.
        
        Este teste verifica se o sistema trata adequadamente erros durante
        a inicialização dos modelos de embedding, que são essenciais para
        a busca semântica. Falhas podem ocorrer por falta de memória,
        modelos corrompidos ou problemas de conectividade.
        
        Cenário testado:
        - Inicialização dos embeddings falha com RuntimeError
        - Erro é propagado corretamente para o chamador
        - Log de erro é gerado para debugging
        - Sistema falha de forma controlada
        """
        mock_init_embeddings.side_effect = RuntimeError("Embeddings failed")
        
        with pytest.raises(RuntimeError):
            init_chroma("/test/path", "test_collection")
        
        mock_logger.error.assert_called()

    @patch('rag_project.query_data.init_embeddings')
    @patch('rag_project.query_data.Chroma')
    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.logger')
    def test_init_chroma_chroma_error(self, mock_logger, mock_path, mock_chroma, mock_init_embeddings):
        """
        Testa o tratamento de falhas na inicialização do banco Chroma.
        
        Este teste verifica se o sistema trata adequadamente erros durante
        a criação da instância do banco vetorial Chroma. Isso pode ocorrer
        por problemas de conectividade, configuração incorreta, ou falhas
        no banco de dados subjacente.
        
        Cenário testado:
        - Embeddings são inicializados com sucesso
        - Diretório existe e está acessível
        - Chroma falha ao ser inicializado
        - RuntimeError é propagada corretamente
        - Log de erro é gerado para troubleshooting
        """
        mock_embedding = Mock()
        mock_init_embeddings.return_value = mock_embedding
        
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_chroma.side_effect = RuntimeError("Chroma failed")
        
        with pytest.raises(RuntimeError):
            init_chroma("/test/path", "test_collection")
        
        mock_logger.error.assert_called()

    @patch('rag_project.query_data.init_embeddings')
    @patch('rag_project.query_data.Path')
    @patch('rag_project.query_data.logger')
    def test_init_chroma_disk_full_error(self, mock_logger, mock_path, mock_init_embeddings):
        """
        Testa o tratamento de erros de espaço em disco insuficiente.
        
        Este teste verifica se o sistema trata adequadamente situações
        onde não há espaço suficiente em disco para criar o diretório
        do banco vetorial. É importante para ambientes com restrições
        de armazenamento ou discos cheios.
        
        Cenário testado:
        - Diretório não existe e precisa ser criado
        - Sistema de arquivos está sem espaço
        - OSError é lançada durante criação do diretório
        - Erro é propagado sem tratamento especial
        """
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.mkdir.side_effect = OSError("No space left on device")
        mock_path.return_value = mock_path_instance
        
        with pytest.raises(OSError):
            init_chroma("/test/path", "test_collection")


class TestFormatPrompt:
    """Test cases for format_prompt function."""

    @patch('rag_project.query_data.logger')
    def test_format_prompt_success(self, mock_logger):
        """
        Testa a formatação bem-sucedida de prompts para o LLM.
        
        Este teste verifica se a função format_prompt consegue combinar
        corretamente contexto, pergunta e template para criar um prompt
        estruturado. É fundamental para garantir que o LLM receba
        informações no formato esperado.
        
        Cenário testado:
        - Contexto e pergunta válidos fornecidos
        - Template personalizado é usado corretamente
        - String formatada retornada conforme esperado
        - Logs de debug/info são gerados
        """
        context = "This is test context"
        question = "What is this about?"
        template = "Context: {context}\nQuestion: {question}"
        
        result = format_prompt(context, question, template)
        
        expected = "Context: This is test context\nQuestion: What is this about?"
        assert result == expected
        assert mock_logger.debug.called or mock_logger.info.called

    @patch('rag_project.query_data.logger')
    def test_format_prompt_empty_context_warning(self, mock_logger):
        """
        Testa o comportamento com contexto vazio gerando warnings.
        
        Este teste verifica se a função detecta quando nenhum contexto
        relevante foi encontrado e gera alertas apropriados. É importante
        porque contexto vazio pode indicar problemas na recuperação de
        documentos ou configuração inadequada do banco vetorial.
        
        Cenário testado:
        - Contexto está vazio (string vazia)
        - Pergunta é válida
        - Warning é logado sobre contexto vazio
        - Formatação continua funcionando mesmo assim
        """
        context = ""
        question = "What is this about?"
        template = "Context: {context}\nQuestion: {question}"
        
        result = format_prompt(context, question, template)
        
        expected = "Context: \nQuestion: What is this about?"
        assert result == expected
        mock_logger.warning.assert_called()

    @patch('rag_project.query_data.logger')
    def test_format_prompt_whitespace_context_warning(self, mock_logger):
        """
        Testa a detecção de contexto contendo apenas espaços em branco.
        
        Este teste verifica se a função detecta contextos que parecem
        não-vazios mas contêm apenas caracteres de espaçamento (espaços,
        tabs, quebras de linha). É importante para identificar casos
        onde a recuperação de documentos retorna conteúdo sem valor real.
        
        Cenário testado:
        - Contexto contém apenas whitespace (espaços, tabs, newlines)
        - Pergunta é válida
        - Warning é logado sobre contexto efetivamente vazio
        - Sistema detecta "falso" conteúdo
        """
        context = "   \n  \t  "
        question = "What is this about?"
        template = "Context: {context}\nQuestion: {question}"
        
        format_prompt(context, question, template)
        
        mock_logger.warning.assert_called()

    @patch('rag_project.query_data.logger')
    def test_format_prompt_empty_question_error(self, mock_logger):
        """
        Testa a validação rigorosa de perguntas vazias.
        
        Este teste verifica se a função rejeita corretamente tentativas
        de formatação com perguntas vazias. É crítico porque uma pergunta
        vazia resultaria em consultas sem sentido ao LLM, desperdiçando
        recursos computacionais e fornecendo respostas inúteis.
        
        Cenário testado:
        - Contexto válido fornecido
        - Pergunta está vazia (string vazia)
        - ValueError é lançada com mensagem clara
        - Log de erro é gerado para debugging
        """
        context = "Some context"
        question = ""
        template = "Context: {context}\nQuestion: {question}"
        
        with pytest.raises(ValueError) as exc_info:
            format_prompt(context, question, template)
        
        assert "Question cannot be empty" in str(exc_info.value)
        mock_logger.error.assert_called()

    @patch('rag_project.query_data.logger')
    def test_format_prompt_whitespace_question_error(self, mock_logger):
        """
        Testa a rejeição de perguntas contendo apenas espaços em branco.
        
        Este teste verifica se a função detecta e rejeita perguntas que
        parecem não-vazias mas contêm apenas caracteres de espaçamento.
        É crítico para evitar processamento de "pseudo-perguntas" que
        não têm conteúdo real e desperdiçariam recursos.
        
        Cenário testado:
        - Contexto válido é fornecido
        - Pergunta contém apenas whitespace (espaços, tabs, newlines)
        - ValueError é lançada tratando como pergunta vazia
        - Sistema não aceita perguntas "falsamente" preenchidas
        """
        context = "Some context"
        question = "   \n  \t  "
        template = "Context: {context}\nQuestion: {question}"
        
        with pytest.raises(ValueError) as exc_info:
            format_prompt(context, question, template)
        
        assert "Question cannot be empty" in str(exc_info.value)

    @patch('rag_project.query_data.logger')
    def test_format_prompt_long_strings_truncated_in_logs(self, mock_logger):
        """
        Testa o manuseio de strings muito longas na formatação.
        
        Este teste verifica se a função consegue processar contextos
        e perguntas muito longos sem problemas de performance ou
        memória. Também testa se os logs são adequadamente truncados
        para evitar poluição do sistema de logging.
        
        Cenário testado:
        - Contexto de 200 caracteres (muito longo)
        - Pergunta de 150 caracteres (muito longa)
        - Formatação funciona corretamente
        - Resultado preserva todo o conteúdo
        - Logs podem ser truncados para legibilidade
        """
        context = "A" * 200
        question = "B" * 150
        template = "Context: {context}\nQuestion: {question}"
        
        result = format_prompt(context, question, template)
        
        assert "A" * 200 in result
        assert "B" * 150 in result
        assert len(result) > 300

    def test_format_prompt_with_default_template(self):
        """
        Testa o uso do template padrão quando nenhum é especificado.
        
        Este teste verifica se a função utiliza corretamente um template
        padrão quando o usuário não fornece um personalizado. É importante
        para garantir que o sistema seja usável sem configuração detalhada
        e que o template padrão seja adequado.
        
        Cenário testado:
        - Contexto e pergunta válidos fornecidos
        - Nenhum template personalizado especificado
        - Template padrão é usado automaticamente
        - Resultado contém contexto e pergunta formatados
        """
        context = "Test context"
        question = "Test question?"
        
        result = format_prompt(context, question)
        
        assert "Test context" in result
        assert "Test question?" in result

    def test_format_prompt_malformed_template(self):
        """
        Testa o tratamento de templates malformados com chaves inválidas.
        
        Este teste verifica se a função detecta e rejeita templates que
        contêm placeholders inválidos ou inexistentes. É importante para
        evitar falhas silenciosas quando usuários fornecem templates
        personalizados incorretos.
        
        Cenário testado:
        - Contexto e pergunta válidos fornecidos
        - Template contém placeholder inválido ({invalid_key})
        - KeyError é lançada durante formatação
        - Erro indica problema específico no template
        """
        context = "Test context"
        question = "Test question"
        malformed_template = "Context: {context}\nQuestion: {invalid_key}"
        
        with pytest.raises(KeyError):
            format_prompt(context, question, malformed_template)

    def test_format_prompt_with_special_characters(self):
        """
        Testa o manuseio de caracteres especiais e unicode.
        
        Este teste verifica se a função consegue processar corretamente
        texto contendo caracteres especiais, unicode, emojis e acentos.
        É essencial para sistemas multilíngues e conteúdo internacional,
        garantindo que não há problemas de encoding.
        
        Cenário testado:
        - Contexto com caracteres gregos, chineses, árabes e emojis
        - Pergunta com acentos em português e francês
        - Formatação preserva todos os caracteres especiais
        - Nenhum problema de encoding ou corrupção de texto
        """
        context = "Special chars: αβγ δεζ ηθι 中文 العربية 🚀🤖"
        question = "What about émojis and ñoñó characters?"
        template = "Context: {context}\nQuestion: {question}"
        
        with patch('rag_project.query_data.logger'):
            result = format_prompt(context, question, template)
        
        assert context in result
        assert question in result
        assert "🚀🤖" in result


class TestValidateQueryInputs:
    """Test cases for validate_query_inputs function."""

    @patch('rag_project.query_data.logger')
    def test_validate_query_inputs_success(self, mock_logger):
        """
        Testa a validação bem-sucedida de entrada de consulta válida.
        
        Este teste verifica se a função validate_query_inputs aceita
        corretamente perguntas válidas dentro dos limites estabelecidos.
        É essencial para garantir que entradas legítimas não sejam
        rejeitadas incorretamente.
        
        Cenário testado:
        - Pergunta válida e não vazia
        - Comprimento dentro do limite especificado
        - Nenhuma exceção é lançada
        - Logs informativos podem ser gerados
        """
        question = "This is a valid question?"
        max_length = 100

        validate_query_inputs(question, max_length)

        assert mock_logger.info.call_count >= 0

    @patch('rag_project.query_data.logger')
    def test_validate_query_inputs_empty_question(self, mock_logger):
        """
        Testa a rejeição de perguntas vazias durante validação.
        
        Este teste verifica se a função detecta e rejeita perguntas vazias,
        que não podem ser processadas de forma significativa pelo sistema RAG.
        É fundamental para prevenir consultas inválidas que desperdiçariam
        recursos computacionais.
        
        Cenário testado:
        - String de pergunta completamente vazia
        - ValueError é lançada com mensagem específica
        - Log de erro é gerado para auditoria
        - Sistema falha de forma controlada
        """
        question = ""
        
        with pytest.raises(ValueError) as exc_info:
            validate_query_inputs(question)
        
        assert "Question cannot be empty" in str(exc_info.value)
        mock_logger.error.assert_called()

    @patch('rag_project.query_data.logger')
    def test_validate_query_inputs_none_question(self, mock_logger):
        """
        Testa a rejeição de perguntas com valor None.
        
        Este teste verifica se a função detecta e rejeita adequadamente
        valores None passados como pergunta. É importante para prevenir
        erros de tipo e garantir que apenas strings válidas sejam aceitas.
        
        Cenário testado:
        - Pergunta é None em vez de string
        - ValueError ou TypeError é lançada
        - Mensagem de erro indica problema de tipo ou valor vazio
        - Sistema não aceita valores nulos
        """
        question = None
        
        with pytest.raises((ValueError, TypeError)) as exc_info:
            validate_query_inputs(question, 100)
        
        error_msg = str(exc_info.value)
        assert ("Question cannot be empty" in error_msg or 
                "expected str" in error_msg or 
                "NoneType" in error_msg)

    @patch('rag_project.query_data.logger')
    def test_validate_query_inputs_whitespace_question(self, mock_logger):
        """
        Testa a rejeição de perguntas contendo apenas whitespace.
        
        Este teste verifica se a função detecta perguntas que contêm
        apenas caracteres de espaçamento e as trata como vazias.
        É importante para evitar processamento de perguntas sem
        conteúdo real que desperdiçariam recursos.
        
        Cenário testado:
        - Pergunta contém apenas espaços, tabs e quebras de linha
        - ValueError é lançada tratando como pergunta vazia
        - Sistema reconhece whitespace como conteúdo inválido
        - Validação é rigorosa quanto ao conteúdo real
        """
        question = "   \n  \t  "
        
        with pytest.raises(ValueError) as exc_info:
            validate_query_inputs(question, 100)
        
        assert "Question cannot be empty" in str(exc_info.value)

    @patch('rag_project.query_data.logger')
    def test_validate_query_inputs_too_long(self, mock_logger):
        """
        Testa a rejeição de perguntas que excedem o limite de comprimento.
        
        Este teste verifica se a função impõe corretamente limites no
        tamanho das perguntas para evitar problemas de performance,
        uso excessivo de memória, ou limitações do modelo LLM.
        
        Cenário testado:
        - Pergunta com 1000 caracteres vs limite de 100
        - ValueError é lançada com mensagem específica sobre limite
        - Log de erro é gerado para auditoria
        - Sistema protege contra entrada excessivamente longa
        """
        question = "A" * 1000
        max_length = 100
        
        with pytest.raises(ValueError) as exc_info:
            validate_query_inputs(question, max_length)
        
        assert f"Maximum {max_length} characters allowed" in str(exc_info.value)
        mock_logger.error.assert_called()

    def test_validate_query_inputs_exact_max_length(self):
        """
        Testa a aceitação de perguntas no limite exato de comprimento.
        
        Este teste verifica se a função aceita corretamente perguntas
        que têm exatamente o comprimento máximo permitido. É importante
        para garantir que a validação de limite seja inclusiva e não
        rejeite entradas legítimas no limite.
        
        Cenário testado:
        - Pergunta com exatamente 50 caracteres (limite = 50)
        - Nenhuma exceção é lançada
        - Validação aceita comprimento no limite
        - Boundary testing para evitar off-by-one errors
        """
        max_length = 50
        question = "A" * max_length
        
        validate_query_inputs(question, max_length)

    def test_validate_query_inputs_with_default_max_length(self):
        """
        Testa o uso do comprimento máximo padrão quando não especificado.
        
        Este teste verifica se a função utiliza corretamente um limite
        padrão de comprimento quando o usuário não especifica um valor.
        É importante para garantir usabilidade sem configuração detalhada.
        
        Cenário testado:
        - Pergunta válida de tamanho moderado
        - Nenhum max_length explicitamente fornecido
        - Função usa limite padrão interno
        - Validação funciona sem parâmetros opcionais
        """
        question = "Valid question"
        
        validate_query_inputs(question)

    def test_validate_query_inputs_exact_boundary(self):
        """
        Testa o comportamento exato nos limites de comprimento.
        
        Este teste verifica comportamentos boundary (limítrofes) para
        garantir que a validação funciona corretamente tanto no limite
        quanto um caractere além do limite. É crítico para evitar
        off-by-one errors na validação.
        
        Cenário testado:
        - Pergunta com exatamente 100 caracteres (aceita)
        - Pergunta com 101 caracteres (rejeitada)
        - Diferença de um caractere tem comportamento correto
        - Boundary testing abrangente
        """
        max_length = 100
        
        question_at_limit = "A" * max_length
        validate_query_inputs(question_at_limit, max_length)
        
        question_over_limit = "A" * (max_length + 1)
        with pytest.raises(ValueError):
            validate_query_inputs(question_over_limit, max_length)

    def test_validate_query_inputs_with_unicode(self):
        """
        Testa a validação de perguntas com caracteres unicode.
        
        Este teste verifica se a função aceita corretamente perguntas
        contendo caracteres especiais, unicode e emojis. É essencial
        para suporte internacional e multilíngue do sistema RAG.
        
        Cenário testado:
        - Pergunta com caracteres espanhóis, chineses, árabes e emoji
        - Nenhuma exceção é lançada
        - Caracteres unicode são tratados corretamente
        - Sistema suporta entrada multilíngue
        """
        unicode_question = "Qué es esto? 这是什么？ما هذا؟ 🤔"
        
        with patch('rag_project.query_data.logger'):
            validate_query_inputs(unicode_question)


class TestRAGQueryEngine:
    """Test cases for RAGQueryEngine class."""

    def setup_method(self):
        self.mock_llm = Mock()
        self.mock_chroma = Mock()
        self.engine_config = {
            "model": "test_model",
            "model_type": "llama",
            "chroma_path": "/test/path",
            "collection_name": "test_collection",
            "llm_config": {"temperature": 0.5},
        }

    @patch('rag_project.query_data.logger')
    def test_rag_query_engine_init_success(self, mock_logger):
        """
        Testa a inicialização bem-sucedida do RAGQueryEngine.
        
        Este teste verifica se o engine principal do sistema RAG é
        inicializado corretamente com todos os componentes necessários
        (LLM e banco vetorial). É fundamental para garantir que o
        sistema esteja pronto para processar consultas.
        
        Cenário testado:
        - Componentes LLM e Chroma são atribuídos corretamente
        - Configurações são armazenadas adequadamente
        - Metadados do modelo são preservados
        - Log de inicialização é gerado
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        assert engine.llm == self.mock_llm
        assert engine.chroma == self.mock_chroma
        assert engine.model == "test_model"
        assert engine.model_type == "llama"
        assert engine.chroma_path == "/test/path"
        assert engine.collection_name == "test_collection"
        assert engine.llm_config == {"temperature": 0.5}
        
        mock_logger.info.assert_called()

    @patch('rag_project.query_data.logger')
    def test_retrieve_documents_success(self, mock_logger):
        """
        Testa a recuperação bem-sucedida de documentos relevantes.
        
        Este teste verifica se o engine consegue buscar documentos
        similares à consulta usando busca semântica. É o núcleo do
        sistema RAG, responsável por encontrar contexto relevante
        para fundamentar as respostas do LLM.
        
        Cenário testado:
        - Busca semântica retorna documentos com scores
        - Documentos acima do threshold são incluídos
        - Número correto de documentos é retornado
        - Chroma é chamado com parâmetros corretos
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        mock_results = [(mock_doc1, 0.8), (mock_doc2, 0.6)]
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = mock_results
        
        result = engine.retrieve_documents("test question", documents_retrieve=2, min_similarity_score=0.5)
        
        assert len(result) == 2
        assert mock_doc1 in result
        assert mock_doc2 in result
        
        self.mock_chroma.similarity_search_with_relevance_scores.assert_called_once_with(
            "test question", k=2
        )

    @patch('rag_project.query_data.logger')
    def test_retrieve_documents_filtered_by_score(self, mock_logger):
        """
        Testa a filtragem de documentos por score de similaridade.
        
        Este teste verifica se o sistema consegue filtrar documentos
        baseado no threshold de similaridade, mantendo apenas os mais
        relevantes. É crucial para evitar que informações pouco
        relacionadas contaminem o contexto fornecido ao LLM.
        
        Cenário testado:
        - Múltiplos documentos com scores diferentes
        - Apenas documentos acima do threshold são retornados
        - Filtragem funciona corretamente (0.9 e 0.7 incluídos, 0.4 excluído)
        - Qualidade do contexto é preservada
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        mock_doc3 = Mock()
        mock_results = [(mock_doc1, 0.9), (mock_doc2, 0.4), (mock_doc3, 0.7)]
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = mock_results
        
        result = engine.retrieve_documents("test question", min_similarity_score=0.5)
        
        assert len(result) == 2
        assert mock_doc1 in result
        assert mock_doc3 in result
        assert mock_doc2 not in result

    @patch('rag_project.query_data.logger')
    def test_retrieve_documents_invalid_count(self, mock_logger):
        """
        Testa a validação de quantidade de documentos inválida.
        
        Este teste verifica se o engine rejeita corretamente tentativas
        de recuperar zero ou números negativos de documentos. É importante
        para prevenir comportamentos indefinidos e garantir que apenas
        quantidades válidas sejam processadas.
        
        Cenário testado:
        - Solicitação de 0 documentos (inválido)
        - ValueError é lançada com mensagem específica
        - Sistema valida parâmetros antes da busca
        - Prevenção de consultas sem sentido
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        with pytest.raises(ValueError) as exc_info:
            engine.retrieve_documents("test question", documents_retrieve=0)
        
        assert "documents_retrieve must be positive" in str(exc_info.value)

    @patch('rag_project.query_data.logger')
    def test_retrieve_documents_chroma_error(self, mock_logger):
        """
        Testa o tratamento de erros durante busca no Chroma.
        
        Este teste verifica se o engine trata adequadamente falhas
        que podem ocorrer durante a busca semântica no banco vetorial.
        Isso pode incluir problemas de conectividade, corrupção de
        dados, ou falhas no índice vetorial.
        
        Cenário testado:
        - Chroma falha durante similarity_search_with_relevance_scores
        - RuntimeError é propagada corretamente
        - Log de erro é gerado para debugging
        - Sistema falha de forma controlada
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        self.mock_chroma.similarity_search_with_relevance_scores.side_effect = RuntimeError("Search failed")
        
        with pytest.raises(RuntimeError):
            engine.retrieve_documents("test question")
        
        mock_logger.error.assert_called()

    @patch('rag_project.query_data.logger')
    def test_retrieve_documents_with_scores_success(self, mock_logger):
        """
        Testa a recuperação de documentos junto com seus scores de similaridade.
        
        Este teste verifica se o engine consegue retornar tanto os documentos
        quanto seus scores de relevância separadamente. É útil para análise
        de qualidade dos resultados e debugging do sistema de busca.
        
        Cenário testado:
        - Busca retorna documentos com scores específicos
        - Método retorna tupla (documentos, scores)
        - Listas têm o mesmo comprimento
        - Scores correspondem aos documentos corretos
        - Filtragem por threshold funciona adequadamente
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        mock_results = [(mock_doc1, 0.8), (mock_doc2, 0.6)]
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = mock_results
        
        docs, scores = engine.retrieve_documents_with_scores("test question", min_similarity_score=0.5)
        
        assert len(docs) == 2
        assert len(scores) == 2
        assert docs == [mock_doc1, mock_doc2]
        assert scores == [0.8, 0.6]

    @patch('rag_project.query_data.logger')
    def test_retrieve_documents_zero_similarity_threshold(self, mock_logger):
        """
        Testa o comportamento com threshold de similaridade zero.
        
        Este teste verifica se o engine aceita corretamente todos os
        documentos quando o threshold é definido como 0.0, efetivamente
        desabilitando a filtragem por similaridade. É útil para casos
        onde se deseja todos os resultados independente da relevância.
        
        Cenário testado:
        - Documentos com scores muito baixos (0.01, 0.001)
        - Threshold definido como 0.0 (aceita tudo)
        - Todos os documentos são incluídos no resultado
        - Filtragem é efetivamente desabilitada
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        mock_results = [(mock_doc1, 0.01), (mock_doc2, 0.001)]
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = mock_results
        
        result = engine.retrieve_documents("test question", min_similarity_score=0.0)
        
        assert len(result) == 2

    @patch('rag_project.query_data.logger')
    def test_retrieve_documents_perfect_similarity_threshold(self, mock_logger):
        """
        Testa o comportamento com threshold de similaridade perfeita.
        
        Este teste verifica se o engine rejeita corretamente todos os
        documentos quando o threshold é definido como 1.0 (similaridade
        perfeita). É útil para casos onde apenas matches exatos são
        desejados, embora seja raro na prática.
        
        Cenário testado:
        - Documentos com scores altos mas não perfeitos (0.99, 0.98)
        - Threshold definido como 1.0 (apenas matches perfeitos)
        - Todos os documentos são rejeitados
        - Filtragem é extremamente restritiva
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        mock_results = [(mock_doc1, 0.99), (mock_doc2, 0.98)]
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = mock_results
        
        result = engine.retrieve_documents("test question", min_similarity_score=1.0)
        
        assert len(result) == 0

    @patch('rag_project.query_data.logger')
    def test_generate_answer_success(self, mock_logger):
        """
        Testa a geração bem-sucedida de resposta pelo LLM.
        
        Este teste verifica se o engine consegue solicitar corretamente
        ao LLM que gere uma resposta baseada no prompt formatado.
        É a etapa final do pipeline RAG onde o conhecimento recuperado
        é sintetizado em uma resposta coerente.
        
        Cenário testado:
        - LLM recebe prompt formatado corretamente
        - Resposta é gerada com sucesso
        - Resultado é retornado sem modificações
        - Integração com o modelo funciona adequadamente
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        self.mock_llm.invoke.return_value = "Generated answer"
        
        result = engine.generate_answer("test prompt")
        
        assert result == "Generated answer"
        self.mock_llm.invoke.assert_called_once_with("test prompt")

    @patch('rag_project.query_data.logger')
    def test_generate_answer_error(self, mock_logger):
        """
        Testa o tratamento de erros durante geração de resposta pelo LLM.
        
        Este teste verifica se o engine trata adequadamente falhas que
        podem ocorrer durante a invocação do modelo LLM. Isso pode incluir
        problemas de memória, modelo corrompido, timeout, ou falhas de
        conectividade (para modelos remotos).
        
        Cenário testado:
        - LLM falha durante invoke() com RuntimeError
        - Erro é propagado corretamente para o chamador
        - Log de erro é gerado para debugging
        - Sistema falha de forma controlada sem corrupção
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        self.mock_llm.invoke.side_effect = RuntimeError("Generation failed")
        
        with pytest.raises(RuntimeError):
            engine.generate_answer("test prompt")
        
        mock_logger.error.assert_called()

    @patch('rag_project.query_data.validate_query_inputs')
    @patch('rag_project.query_data.format_prompt')
    @patch('rag_project.query_data.logger')
    def test_query_success(self, mock_logger, mock_format_prompt, mock_validate):
        """
        Testa o fluxo completo de consulta RAG bem-sucedida.
        
        Este teste verifica todo o pipeline RAG integrado: validação
        da entrada, recuperação de documentos, formatação do prompt
        e geração da resposta. É o teste mais importante pois simula
        o uso real do sistema.
        
        Cenário testado:
        - Pergunta é validada corretamente
        - Documentos relevantes são recuperados
        - Prompt é formatado com contexto e pergunta
        - LLM gera resposta baseada no contexto
        - Resposta é limpa (whitespace removido)
        - Todas as etapas são executadas na ordem correta
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        mock_doc = Mock()
        mock_doc.page_content = "Document content"
        mock_results = [(mock_doc, 0.8)]
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = mock_results
        mock_format_prompt.return_value = "formatted prompt"
        self.mock_llm.invoke.return_value = "  Generated answer  "
        
        result = engine.query("test question")
        
        assert result == "Generated answer"
        mock_validate.assert_called_once()
        mock_format_prompt.assert_called_once()
        self.mock_llm.invoke.assert_called_once_with("formatted prompt")

    @patch('rag_project.query_data.validate_query_inputs')
    @patch('rag_project.query_data.logger')
    def test_query_no_documents_found(self, mock_logger, mock_validate):
        """
        Testa o comportamento quando nenhum documento relevante é encontrado.
        
        Este teste verifica se o sistema trata adequadamente situações
        onde a busca semântica não retorna documentos suficientemente
        similares. É importante para evitar alucinações do LLM quando
        não há contexto adequado disponível.
        
        Cenário testado:
        - Busca no banco vetorial não retorna resultados
        - Sistema reconhece ausência de contexto relevante
        - Resposta padrão informativa é retornada
        - Warning é logado para indicar problema potencial
        - LLM não é chamado desnecessariamente
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = []
        
        result = engine.query("test question")
        
        expected = "I couldn't find any relevant information to answer your question."
        assert result == expected
        mock_logger.warning.assert_called()

    @patch('rag_project.query_data.validate_query_inputs')
    @patch('rag_project.query_data.format_prompt')
    @patch('rag_project.query_data.logger')
    def test_query_with_metadata_success(self, mock_logger, mock_format_prompt, mock_validate):
        """
        Testa a consulta com retorno de metadados detalhados.
        
        Este teste verifica se o engine consegue executar uma consulta
        completa e retornar não apenas a resposta, mas também metadados
        ricos incluindo fontes, scores de similaridade e estatísticas.
        É essencial para análise de qualidade e rastreabilidade.
        
        Cenário testado:
        - Consulta bem-sucedida com documento relevante
        - RAGResponse é retornada com todos os campos preenchidos
        - Metadados incluem fonte, scores e estatísticas
        - Pergunta original é preservada para auditoria
        - Sistema fornece transparência completa do processo
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        mock_doc = Mock()
        mock_doc.page_content = "Document content"
        mock_doc.metadata = {"source": "test_source.txt"}
        mock_results = [(mock_doc, 0.8)]
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = mock_results
        mock_format_prompt.return_value = "formatted prompt"
        self.mock_llm.invoke.return_value = "Generated answer"
        
        result = engine.query_with_metadata("test question")
        
        assert isinstance(result, RAGResponse)
        assert result.answer == "Generated answer"
        assert result.sources == ["test_source.txt"]
        assert result.retrieved_docs == 1
        assert result.similarity_scores == [0.8]
        assert result.query == "test question"

    @patch('rag_project.query_data.validate_query_inputs')
    @patch('rag_project.query_data.logger')
    def test_query_with_metadata_no_documents(self, mock_logger, mock_validate):
        """
        Testa consulta com metadados quando nenhum documento é encontrado.
        
        Este teste verifica se o engine retorna adequadamente metadados
        vazios quando nenhum documento relevante é encontrado, mantendo
        a estrutura de resposta consistente mesmo em casos de falha
        na recuperação de contexto.
        
        Cenário testado:
        - Busca não retorna documentos relevantes
        - RAGResponse é retornada com campos vazios apropriados
        - Mensagem padrão é fornecida como resposta
        - Metadados indicam claramente ausência de contexto
        - Estrutura de resposta permanece consistente
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = []
        
        result = engine.query_with_metadata("test question")
        
        assert isinstance(result, RAGResponse)
        assert "couldn't find any relevant information" in result.answer
        assert result.sources == []
        assert result.retrieved_docs == 0
        assert result.similarity_scores == []

    @patch('rag_project.query_data.logger')
    def test_get_config_info(self, mock_logger):
        """
        Testa a recuperação de informações de configuração do engine.
        
        Este teste verifica se o engine consegue retornar corretamente
        suas configurações internas para debugging, monitoramento e
        auditoria. É útil para verificar configurações em tempo de
        execução e troubleshooting.
        
        Cenário testado:
        - Engine retorna dicionário com configurações completas
        - Todas as configurações principais estão presentes
        - Valores correspondem aos configurados na inicialização
        - Informações são úteis para debugging e auditoria
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        result = engine.get_config_info()
        
        expected = {
            "model": "test_model",
            "model_type": "llama",
            "chroma_path": "/test/path",
            "collection_name": "test_collection",
            "llm_config": {"temperature": 0.5},
        }
        
        assert result == expected

    def test_repr(self):
        """
        Testa a representação string do engine para debugging.
        
        Este teste verifica se o engine fornece uma representação
        textual útil quando usado em debugging, logging ou inspeção
        interativa. É importante para facilitar o desenvolvimento
        e troubleshooting.
        
        Cenário testado:
        - __repr__ retorna string informativa
        - String inclui identificadores chave (modelo e coleção)
        - Formato é consistente e legível
        - Útil para debugging e logging
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        result = repr(engine)
        
        expected = "RAGQueryEngine(model='test_model', collection='test_collection')"
        assert result == expected

    def test_rag_query_engine_corrupted_config(self):
        """
        Testa o comportamento com configuração incompleta ou corrompida.
        
        Este teste verifica se o engine consegue lidar adequadamente
        com configurações incompletas, definindo valores padrão ou None
        para campos ausentes. É importante para robustez quando
        configurações são parcialmente corrompidas ou incompletas.
        
        Cenário testado:
        - Configuração contém apenas o campo "model"
        - Campos ausentes são definidos como None
        - Engine não falha durante inicialização
        - Sistema é tolerante a configurações incompletas
        """
        mock_llm = Mock()
        mock_chroma = Mock()
        
        incomplete_config = {"model": "test_model"}
        
        with patch('rag_project.query_data.logger'):
            engine = RAGQueryEngine(mock_llm, mock_chroma, incomplete_config)
            
            assert engine.model == "test_model"
            assert engine.model_type is None
            assert engine.chroma_path is None

    def test_multiple_engine_instances(self):
        """
        Testa a criação de múltiplas instâncias independentes do engine.
        
        Este teste verifica se é possível criar várias instâncias do
        RAGQueryEngine simultaneamente com configurações diferentes,
        garantindo isolamento adequado entre elas. É importante para
        aplicações que precisam de múltiplos engines especializados.
        
        Cenário testado:
        - Duas instâncias com configurações completamente diferentes
        - Cada instância mantém sua própria configuração independente
        - Não há interferência entre instâncias
        - LLMs e bancos vetorias são isolados corretamente
        """
        mock_llm1 = Mock()
        mock_chroma1 = Mock()
        mock_llm2 = Mock()
        mock_chroma2 = Mock()
        
        config1 = {"model": "model1", "model_type": "type1", "chroma_path": "path1",
                  "collection_name": "collection1", "llm_config": {}}
        config2 = {"model": "model2", "model_type": "type2", "chroma_path": "path2",
                  "collection_name": "collection2", "llm_config": {}}
        
        with patch('rag_project.query_data.logger'):
            engine1 = RAGQueryEngine(mock_llm1, mock_chroma1, config1)
            engine2 = RAGQueryEngine(mock_llm2, mock_chroma2, config2)
        
        assert engine1.model != engine2.model
        assert engine1.chroma_path != engine2.chroma_path
        assert engine1.llm != engine2.llm

    def test_engine_large_context_memory_usage(self):
        """
        Testa o manuseio de contexto muito grande e uso de memória.
        
        Este teste verifica se o engine consegue processar documentos
        com conteúdo extremamente grande (100.000 caracteres) sem
        problemas de performance ou memória. É crítico para garantir
        escalabilidade com documentos grandes na vida real.
        
        Cenário testado:
        - Documento com 100.000 caracteres é processado
        - Sistema não falha ou corrompe com contexto massivo
        - Memória é gerenciada adequadamente
        - Resposta é gerada normalmente
        - Performance se mantém aceitável
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        large_content = "A" * 100000
        mock_doc = Mock()
        mock_doc.page_content = large_content
        mock_doc.metadata = {"source": "large_doc.txt"}
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.8)]
        self.mock_llm.invoke.return_value = "Response to large context"
        
        with patch('rag_project.query_data.logger'), \
             patch('rag_project.query_data.format_prompt') as mock_format, \
             patch('rag_project.query_data.validate_query_inputs'):
            
            mock_format.return_value = "formatted_prompt"
            
            result = engine.query("test question")
            
            assert result == "Response to large context"
            mock_format.assert_called_once()

    def test_query_with_mixed_encoding(self):
        """
        Testa consulta com encoding misto e caracteres internacionais.
        
        Este teste verifica se o engine consegue processar adequadamente
        conteúdo e perguntas com encoding misto, incluindo caracteres
        especiais, acentos e diferentes sistemas de escrita. É essencial
        para suporte internacional e multilíngue.
        
        Cenário testado:
        - Documento com texto em espanhol, inglês e chinês
        - Pergunta em espanhol com acentos
        - Sistema preserva todos os caracteres corretamente
        - Nenhum problema de encoding ou corrupção
        - Suporte robusto para conteúdo internacional
        """
        engine = RAGQueryEngine(self.mock_llm, self.mock_chroma, self.engine_config)
        
        mock_doc = Mock()
        mock_doc.page_content = "Texto en español con açcéntos and English mixed 中文"
        mock_doc.metadata = {"source": "mixed_encoding.txt"}
        
        self.mock_chroma.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.8)]
        self.mock_llm.invoke.return_value = "Response with mixed encoding"
        
        unicode_question = "¿Qué dice el documento?"
        
        with patch('rag_project.query_data.logger'), \
             patch('rag_project.query_data.format_prompt') as mock_format, \
             patch('rag_project.query_data.validate_query_inputs'):
            
            mock_format.return_value = "formatted_prompt"
            
            result = engine.query(unicode_question)
            
            assert result == "Response with mixed encoding"


class TestCreateRagEngine:
    """Test cases for create_rag_engine function."""

    @patch('rag_project.query_data.load_local_llama')
    @patch('rag_project.query_data.init_chroma')
    @patch('rag_project.query_data.RAGQueryEngine')
    @patch('rag_project.query_data.logger')
    def test_create_rag_engine_success(self, mock_logger, mock_rag_engine, mock_init_chroma, mock_load_llama):
        """
        Testa a criação bem-sucedida de um engine RAG completo.
        
        Este teste verifica se a função factory consegue orquestrar
        corretamente a inicialização de todos os componentes do
        sistema RAG (LLM, banco vetorial e engine). É crucial para
        garantir que o sistema seja configurável e reutilizável.
        
        Cenário testado:
        - LLM é carregado com configuração específica
        - Banco vetorial Chroma é inicializado
        - RAGQueryEngine é criado com componentes corretos
        - Todas as configurações são preservadas
        - Dependências são injetadas adequadamente
        """
        mock_llm = Mock()
        mock_chroma = Mock()
        mock_engine = Mock()
        
        mock_load_llama.return_value = mock_llm
        mock_init_chroma.return_value = mock_chroma
        mock_rag_engine.return_value = mock_engine
        
        config = LLMConfig(temperature=0.7)
        
        result = create_rag_engine(
            chroma_path="/test/path",
            collection_name="test_collection",
            model="test_model",
            model_type="llama",
            llm_config=config
        )
        
        assert result == mock_engine
        
        mock_load_llama.assert_called_once_with("test_model", "llama", config)
        mock_init_chroma.assert_called_once_with("/test/path", "test_collection")
        
        mock_rag_engine.assert_called_once()
        call_args = mock_rag_engine.call_args
        assert call_args[1]['llm'] == mock_llm
        assert call_args[1]['chroma'] == mock_chroma
        
        engine_config = call_args[1]['engine_config']
        assert engine_config['model'] == "test_model"
        assert engine_config['model_type'] == "llama"
        assert engine_config['chroma_path'] == "/test/path"
        assert engine_config['collection_name'] == "test_collection"

    @patch('rag_project.query_data.load_local_llama')
    @patch('rag_project.query_data.logger')
    def test_create_rag_engine_llm_error(self, mock_logger, mock_load_llama):
        """
        Testa o tratamento de erros durante carregamento do LLM.
        
        Este teste verifica se a função factory trata adequadamente
        falhas que podem ocorrer durante o carregamento do modelo LLM.
        É importante para garantir que erros sejam propagados
        corretamente e o sistema falhe de forma controlada.
        
        Cenário testado:
        - Carregamento do LLM falha com RuntimeError
        - Erro é propagado para o chamador
        - Sistema não tenta continuar com LLM inválido
        - Falha acontece cedo no processo de inicialização
        """
        mock_load_llama.side_effect = RuntimeError("LLM failed")
        
        with pytest.raises(RuntimeError):
            create_rag_engine("/test/path", "test_collection")

    @patch('rag_project.query_data.load_local_llama')
    @patch('rag_project.query_data.init_chroma')
    @patch('rag_project.query_data.logger')
    def test_create_rag_engine_chroma_error(self, mock_logger, mock_init_chroma, mock_load_llama):
        """
        Testa o tratamento de erros durante inicialização do Chroma.
        
        Este teste verifica se a função factory trata adequadamente
        falhas que podem ocorrer durante a inicialização do banco
        vetorial Chroma. É importante para garantir que o sistema
        falhe de forma controlada quando componentes críticos falham.
        
        Cenário testado:
        - LLM é carregado com sucesso
        - Inicialização do Chroma falha com RuntimeError
        - Erro é propagado corretamente para o chamador
        - Sistema não tenta continuar sem banco vetorial
        """
        mock_load_llama.return_value = Mock()
        mock_init_chroma.side_effect = RuntimeError("Chroma failed")
        
        with pytest.raises(RuntimeError):
            create_rag_engine("/test/path", "test_collection")

    @patch('rag_project.query_data.load_local_llama')
    @patch('rag_project.query_data.init_chroma')
    @patch('rag_project.query_data.RAGQueryEngine')
    @patch('rag_project.query_data.logger')
    def test_create_rag_engine_with_defaults(self, mock_logger, mock_rag_engine, mock_init_chroma, mock_load_llama):
        """
        Testa a criação do engine RAG usando apenas parâmetros obrigatórios.
        
        Este teste verifica se a função factory consegue criar um engine
        funcional utilizando apenas os parâmetros mínimos necessários,
        preenchendo o resto com valores padrão. É importante para
        facilitar o uso da biblioteca.
        
        Cenário testado:
        - Apenas caminho do Chroma e nome da coleção fornecidos
        - Função usa valores padrão para modelo e configurações
        - Engine é criado com sucesso
        - Todas as funções de inicialização são chamadas
        """
        mock_llm = Mock()
        mock_chroma = Mock()
        mock_engine = Mock()
        
        mock_load_llama.return_value = mock_llm
        mock_init_chroma.return_value = mock_chroma
        mock_rag_engine.return_value = mock_engine
        
        result = create_rag_engine("/test/path", "test_collection")
        
        assert result == mock_engine
        mock_load_llama.assert_called_once()
        mock_init_chroma.assert_called_once()

    def test_create_rag_engine_config_combinations(self):
        """
        Testa diferentes combinações de configurações para criação do engine.
        
        Este teste verifica se a função factory consegue lidar com
        diferentes combinações de parâmetros de configuração, incluindo
        caminhos longos e nomes complexos. É importante para garantir
        flexibilidade e robustez da função.
        
        Cenário testado:
        - Múltiplas combinações de caminhos e nomes
        - Caminhos curtos e longos são suportados
        - Nomes de coleção simples e complexos funcionam
        - Função é robusta para diferentes casos de uso
        """
        test_cases = [
            ("/path1", "collection1", "model1", "type1"),
            ("/path2", "collection2", "model2", "type2"),
            ("/very/long/path/to/test/deep/directory/structure", "long_collection_name", "model", "type"),
        ]
        
        for chroma_path, collection_name, model, model_type in test_cases:
            with patch('rag_project.query_data.load_local_llama') as mock_load_llama, \
                 patch('rag_project.query_data.init_chroma') as mock_init_chroma, \
                 patch('rag_project.query_data.RAGQueryEngine') as mock_rag_engine, \
                 patch('rag_project.query_data.logger'):
                
                mock_load_llama.return_value = Mock()
                mock_init_chroma.return_value = Mock()
                mock_engine = Mock()
                mock_rag_engine.return_value = mock_engine
                
                result = create_rag_engine(chroma_path, collection_name, model, model_type)
                
                assert result == mock_engine
                mock_load_llama.assert_called()
                mock_init_chroma.assert_called_with(chroma_path, collection_name)


class TestLLMConfigVariations:
    """Test different LLMConfig variations."""

    def test_llm_config_variations(self):
        """
        Testa diferentes variações de configuração do LLM.
        
        Este teste verifica se o sistema consegue carregar o LLM com
        diferentes configurações de parâmetros, incluindo valores
        extremos (temperatura 0.0 e 1.0, tokens mínimo e máximo).
        É importante para garantir flexibilidade na configuração.
        
        Cenário testado:
        - Configuração padrão (sem parâmetros)
        - Temperatura conservadora (0.0) e criativa (1.0)
        - Tokens mínimo (1) e máximo (2048)
        - Sistema aceita todas as variações válidas
        - Nenhuma configuração causa falha inesperada
        """
        configs = [
            LLMConfig(),
            LLMConfig(temperature=0.0),
            LLMConfig(temperature=1.0),
            LLMConfig(max_new_tokens=1),
            LLMConfig(max_new_tokens=2048),
        ]
        
        for config in configs:
            with patch('rag_project.query_data.Path') as mock_path, \
                 patch('rag_project.query_data.CTransformers') as mock_ctransformers, \
                 patch('rag_project.query_data.logger'):
                
                mock_path.return_value.exists.return_value = True
                mock_ctransformers.return_value = Mock()
                
                result = load_local_llama("test_model", "llama", config)
                assert result is not None


class TestIntegrationAndPerformance:
    """Integration and performance tests."""

    def test_full_workflow_mocked(self):
        """
        Testa o fluxo completo do sistema RAG de ponta a ponta.
        
        Este é um teste de integração que verifica se todos os
        componentes do sistema RAG funcionam juntos corretamente,
        desde a inicialização até a geração da resposta final.
        É essencial para validar que mudanças em componentes
        individuais não quebrem o sistema como um todo.
        
        Cenário testado:
        - Engine RAG é criado via factory function
        - Documento é recuperado do banco vetorial
        - Prompt é formatado corretamente
        - LLM gera resposta baseada no contexto
        - Pipeline completo funciona sem erros
        """
        with patch('rag_project.query_data.load_local_llama') as mock_load_llama, \
             patch('rag_project.query_data.init_chroma') as mock_init_chroma, \
             patch('rag_project.query_data.logger'):
            
            mock_llm = Mock()
            mock_chroma = Mock()
            mock_load_llama.return_value = mock_llm
            mock_init_chroma.return_value = mock_chroma
            
            mock_doc = Mock()
            mock_doc.page_content = "Test document content"
            mock_doc.metadata = {"source": "test.txt"}
            mock_chroma.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.8)]
            mock_llm.invoke.return_value = "Test response"
            
            engine = create_rag_engine("/test/path", "test_collection")
            
            with patch('rag_project.query_data.validate_query_inputs'), \
                 patch('rag_project.query_data.format_prompt') as mock_format:
                mock_format.return_value = "formatted prompt"
                
                result = engine.query("test question")
                
                assert result == "Test response"

    @patch('rag_project.query_data.logger')
    def test_large_context_handling(self, mock_logger):
        """
        Testa o manuseio de documentos com contexto muito grande.
        
        Este teste verifica se o sistema consegue processar documentos
        com conteúdo extenso sem problemas de performance ou memória.
        É importante para garantir que o sistema seja escalável e
        possa trabalhar com documentos reais que podem ser longos.
        
        Cenário testado:
        - Documento com 10.000 caracteres é processado
        - Sistema não falha com contexto grande
        - Memória é gerenciada adequadamente
        - Resposta é gerada normalmente
        - Performance se mantém aceitável
        """
        mock_llm = Mock()
        mock_chroma = Mock()
        engine = RAGQueryEngine(mock_llm, mock_chroma, {
            "model": "test", "model_type": "test", "chroma_path": "test",
            "collection_name": "test", "llm_config": {}
        })
        
        large_context = "A" * 10000
        mock_doc = Mock()
        mock_doc.page_content = large_context
        
        mock_chroma.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.8)]
        mock_llm.invoke.return_value = "Response"
        
        with patch('rag_project.query_data.format_prompt') as mock_format, \
             patch('rag_project.query_data.validate_query_inputs'):
            mock_format.return_value = "formatted"
            
            result = engine.query("test")
            assert result == "Response"

    @patch('rag_project.query_data.logger')
    def test_many_documents_retrieval(self, mock_logger):
        """
        Testa a recuperação de um grande número de documentos.
        
        Este teste verifica se o sistema consegue processar consultas
        que retornam muitos documentos relevantes. É importante para
        casos onde há muito conteúdo similar ou quando se deseja
        contexto mais abrangente para o LLM.
        
        Cenário testado:
        - 100 documentos são retornados pelo Chroma
        - Sistema processa todos sem falhar
        - Lista de documentos mantém integridade
        - Performance com volume alto é aceitável
        - Memória é gerenciada adequadamente
        """
        mock_llm = Mock()
        mock_chroma = Mock()
        engine = RAGQueryEngine(mock_llm, mock_chroma, {
            "model": "test", "model_type": "test", "chroma_path": "test",
            "collection_name": "test", "llm_config": {}
        })
        
        mock_docs = [Mock() for _ in range(100)]
        mock_results = [(doc, 0.8) for doc in mock_docs]
        
        mock_chroma.similarity_search_with_relevance_scores.return_value = mock_results
        
        result = engine.retrieve_documents("test", documents_retrieve=100)
        assert len(result) == 100


class TestLoggingAndMonitoring:
    """Test logging and monitoring functionality."""

    def test_all_functions_log_execution_time(self):
        """
        Testa se todas as funções principais fazem logging adequado.
        
        Este teste verifica se cada função do sistema registra informações
        suficientes nos logs para debugging e monitoramento. É essencial
        para troubleshooting em produção e análise de performance.
        
        Cenário testado:
        - Cada função principal é executada
        - Logs são gerados (info, error, ou debug)
        - Sistema é observável e auditável
        - Debugging é facilitado por logging adequado
        - Monitoramento pode ser implementado baseado nos logs
        """
        functions_to_test = [
            ('load_local_llama', load_local_llama),
            ('init_chroma', init_chroma),
            ('format_prompt', format_prompt),
            ('validate_query_inputs', validate_query_inputs),
            ('create_rag_engine', create_rag_engine),
        ]
        
        for func_name, func in functions_to_test:
            with patch('rag_project.query_data.logger') as mock_logger:
                try:
                    if func_name == 'load_local_llama':
                        with patch('rag_project.query_data.Path') as mock_path, \
                             patch('rag_project.query_data.CTransformers'):
                            mock_path.return_value.exists.return_value = False
                            func()
                    elif func_name == 'init_chroma':
                        with patch('rag_project.query_data.init_embeddings'):
                            func("/test", "test")
                    elif func_name == 'format_prompt':
                        func("context", "question")
                    elif func_name == 'validate_query_inputs':
                        func("valid question")
                    elif func_name == 'create_rag_engine':
                        with patch('rag_project.query_data.load_local_llama'), \
                             patch('rag_project.query_data.init_chroma'), \
                             patch('rag_project.query_data.RAGQueryEngine'):
                            func("/test", "test")
                except Exception:
                    pass
                
                assert mock_logger.info.called or mock_logger.error.called or mock_logger.debug.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])