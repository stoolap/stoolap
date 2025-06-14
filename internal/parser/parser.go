/*
Copyright 2025 Stoolap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package parser

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/stoolap/stoolap/internal/functions/registry"
	"github.com/stoolap/stoolap/internal/parser/funcregistry"

	_ "github.com/stoolap/stoolap/internal/functions/aggregate"
	_ "github.com/stoolap/stoolap/internal/functions/scalar"
	_ "github.com/stoolap/stoolap/internal/functions/window"
)

// Precedence levels for operators
const (
	_ int = iota
	LOWEST
	LOGICAL     // AND, OR
	EQUALS      // =, <>, !=
	LESSGREATER // >, <, >=, <=
	SUM         // +, -
	PRODUCT     // *, /, %
	PREFIX      // -X, NOT X
	CALL        // func(X)
	INDEX       // array[index]
	DOT         // table.column
)

// Precedence map maps operators to their precedence
var precedences = map[string]int{
	"OR":      LOGICAL,
	"AND":     LOGICAL,
	"=":       EQUALS,
	"<>":      EQUALS,
	"!=":      EQUALS,
	"<":       LESSGREATER,
	">":       LESSGREATER,
	"<=":      LESSGREATER,
	">=":      LESSGREATER,
	"+":       SUM,
	"-":       SUM,
	"||":      SUM,     // String concatenation
	"*":       PRODUCT, // Multiplication
	"/":       PRODUCT,
	"%":       PRODUCT,
	".":       DOT,
	"AS":      LOWEST, // Assign LOWEST precedence to AS keyword
	"LIKE":    EQUALS, // LIKE has same precedence as equals
	"IN":      EQUALS, // IN has same precedence as equals
	"NOT":     EQUALS, // NOT has same precedence as equals
	"BETWEEN": EQUALS, // BETWEEN has same precedence as equals
	"IS":      EQUALS, // IS has same precedence as equals
	"IS NOT":  EQUALS, // IS NOT has same precedence as equals
}

// Parser represents a SQL parser
type Parser struct {
	lexer *Lexer

	curToken  Token
	peekToken Token

	errors []*ParseError // Using our new ParseError type
	query  string        // Store the original query for error context

	// Map of functions for parsing prefix expressions
	prefixParseFns map[TokenType]prefixParseFn
	// Map of functions for parsing infix expressions
	infixParseFns map[TokenType]infixParseFn

	// Function registry for validating function calls
	funcRegistry funcregistry.Registry
	// Function validator for validating function calls
	funcValidator *funcregistry.Validator

	// Parameter tracking for enhanced binding
	currentStatementID        int    // ID of the current statement being parsed
	parameterStatementCounter int    // Counter for parameters within the current statement
	parameterCounter          int    // Counter for parameters across all statements
	currentClause             string // Current clause being parsed (WHERE, SELECT, etc.)
}

// Function type for parsing prefix expressions
type prefixParseFn func() Expression

// Function type for parsing infix expressions
type infixParseFn func(Expression) Expression

// NewParser creates a new parser from a lexer
func NewParser(lexer *Lexer) *Parser {
	// Use the global registry from the registry package
	return NewParserWithRegistry(lexer, registry.GetGlobal().ParserRegistry())
}

// NewParserWithRegistry creates a new parser with a custom function registry
func NewParserWithRegistry(lexer *Lexer, registry funcregistry.Registry) *Parser {
	// Create a new parser first
	p := &Parser{
		lexer:                     lexer,
		errors:                    []*ParseError{},
		query:                     lexer.input, // Store the original query for context
		funcRegistry:              registry,
		funcValidator:             funcregistry.NewValidator(registry),
		currentStatementID:        1,         // Start with statement ID 1
		parameterStatementCounter: 0,         // Start with statement parameter counter 0
		parameterCounter:          0,         // Start with parameter counter 0
		currentClause:             "UNKNOWN", // Default clause
	}

	// Initialize prefix parse functions
	p.prefixParseFns = make(map[TokenType]prefixParseFn)
	p.registerPrefix(TokenIdentifier, p.parseIdentifier)
	p.registerPrefix(TokenInteger, p.parseIntegerLiteral)
	p.registerPrefix(TokenFloat, p.parseFloatLiteral)
	p.registerPrefix(TokenString, p.parseStringLiteral)
	p.registerPrefix(TokenKeyword, p.parseKeyword)
	p.registerPrefix(TokenOperator, p.parsePrefixExpression)
	p.registerPrefix(TokenPunctuator, p.parsePunctuator)
	p.registerPrefix(TokenParameter, p.parseParameter)

	// Initialize infix parse functions
	p.infixParseFns = make(map[TokenType]infixParseFn)
	p.registerInfix(TokenOperator, p.parseInfixExpression)
	p.registerInfix(TokenKeyword, p.parseInfixKeyword)
	p.registerInfix(TokenPunctuator, p.parseInfixPunctuator)

	// Read two tokens to initialize curToken and peekToken
	p.nextToken()
	p.nextToken()

	return p
}

// Register a prefix parse function
func (p *Parser) registerPrefix(tokenType TokenType, fn prefixParseFn) {
	p.prefixParseFns[tokenType] = fn
}

// Register an infix parse function
func (p *Parser) registerInfix(tokenType TokenType, fn infixParseFn) {
	p.infixParseFns[tokenType] = fn
}

// Errors returns the list of errors that occurred during parsing
func (p *Parser) Errors() []string {
	// Convert ParseError objects to strings for backward compatibility
	result := make([]string, len(p.errors))
	for i, err := range p.errors {
		result[i] = err.Error()
	}
	slices.Reverse(result)
	return result
}

// ErrorsWithContext returns detailed errors with SQL context
func (p *Parser) ErrorsWithContext() []*ParseError {
	return p.errors
}

// FormatErrors returns a user-friendly formatted error message
func (p *Parser) FormatErrors() string {
	return FormatErrors(p.query, p.errors)
}

// peekError adds an error when the next token isn't the expected one
func (p *Parser) peekError(expected string) {
	msg := fmt.Sprintf("expected next token to be %s, got %s",
		expected, p.peekToken.Type)

	parseErr := &ParseError{
		Message:  msg,
		Position: p.peekToken.Position,
		Context:  p.query,
	}
	p.errors = append(p.errors, parseErr)
}

// addError adds a generic error
func (p *Parser) addError(msg string) {
	// Use current token position for context
	parseErr := &ParseError{
		Message:  msg,
		Position: p.curToken.Position,
		Context:  p.query,
	}
	p.errors = append(p.errors, parseErr)
}

// tokenDebug returns a debug representation of a token
func (p *Parser) tokenDebug(token Token) string {
	return fmt.Sprintf("%s(%s) at %s", token.Type, token.Literal, token.Position)
}

// nextToken advances to the next token
func (p *Parser) nextToken() {
	p.curToken = p.peekToken
	p.peekToken = p.lexer.NextToken()
}

// curTokenIs checks if the current token is of the given type
func (p *Parser) curTokenIs(t TokenType) bool {
	return p.curToken.Type == t
}

// peekTokenIs checks if the next token is of the given type
func (p *Parser) peekTokenIs(t TokenType) bool {
	return p.peekToken.Type == t
}

// expectPeek checks if the next token is of the given type and advances if true
func (p *Parser) expectPeek(t TokenType) bool {
	if p.peekTokenIs(t) {
		p.nextToken()
		return true
	} else {
		p.peekError(t.String())
		return false
	}
}

// expectKeyword checks if the next token is the specified keyword and advances if true
func (p *Parser) expectKeyword(keyword string) bool {
	if p.peekTokenIs(TokenKeyword) && strings.EqualFold(p.peekToken.Literal, keyword) {
		p.nextToken()
		return true
	} else {
		p.peekError(fmt.Sprintf("keyword %s", keyword))
		return false
	}
}

// curTokenIsKeyword checks if current token is a specific keyword
func (p *Parser) curTokenIsKeyword(keyword string) bool {
	return p.curTokenIs(TokenKeyword) && strings.EqualFold(p.curToken.Literal, keyword)
}

// peekTokenIsKeyword checks if next token is a specific keyword
func (p *Parser) peekTokenIsKeyword(keyword string) bool {
	return p.peekTokenIs(TokenKeyword) && strings.EqualFold(p.peekToken.Literal, keyword)
}

// peekTokenIsOperator checks if next token is a specific operator
func (p *Parser) peekTokenIsOperator(operator string) bool {
	return p.peekTokenIs(TokenOperator) && p.peekToken.Literal == operator
}

// curTokenIsPunctuator checks if current token is a specific punctuator
func (p *Parser) curTokenIsPunctuator(punctuator string) bool {
	return p.curTokenIs(TokenPunctuator) && p.curToken.Literal == punctuator
}

// peekTokenIsPunctuator checks if next token is a specific punctuator
func (p *Parser) peekTokenIsPunctuator(punctuator string) bool {
	return p.peekTokenIs(TokenPunctuator) && p.peekToken.Literal == punctuator
}

// ParseProgram parses a complete SQL program
func (p *Parser) ParseProgram() *Program {
	program := &Program{
		Statements: []Statement{},
	}

	for !p.curTokenIs(TokenEOF) {
		// Skip any leading semicolons (empty statements)
		for p.curTokenIsPunctuator(";") {
			p.nextToken()
			if p.curTokenIs(TokenEOF) {
				break
			}
		}

		// Parse a statement
		if !p.curTokenIs(TokenEOF) {
			stmt := p.parseStatement()
			if stmt != nil {
				program.Statements = append(program.Statements, stmt)
			}

			// Handle statement termination
			// If next token is a semicolon, consume it and continue
			// Otherwise, the statement should end at EOF
			if p.peekTokenIsPunctuator(";") {
				p.nextToken() // Consume the semicolon
			} /* else if !p.peekTokenIs(TokenEOF) {
				// If not EOF and not semicolon, something is wrong
				// unless we're at the end of the program
				if !p.curTokenIs(TokenEOF) {
					p.addError("expected ';' at end of statement")
				}
			} */

			p.nextToken() // Move to the next token after semicolon or to EOF
		}
	}

	// Validate function calls in the program
	p.validateProgramFunctions(program)

	return program
}

// validateProgramFunctions validates all function calls in the program
func (p *Parser) validateProgramFunctions(program *Program) {
	if program == nil {
		return
	}

	// Traverse each statement and validate function calls
	for _, stmt := range program.Statements {
		// Skip nil statements to avoid panics
		if stmt == nil {
			continue
		}

		// Attempt to validate function calls, but catch any panics
		func() {
			defer func() {
				if r := recover(); r != nil {
					// Log the panic but continue processing other statements
					p.addError(fmt.Sprintf("panic during validation: %v", r))
				}
			}()

			p.validateStatementFunctions(stmt)
		}()
	}
}

// validateStatementFunctions validates function calls in a statement
func (p *Parser) validateStatementFunctions(stmt Statement) {
	if stmt == nil {
		return
	}

	switch s := stmt.(type) {
	case *SelectStatement:
		// Validate WITH clause
		if s.With != nil {
			for _, cte := range s.With.CTEs {
				if cte.Query != nil {
					p.validateStatementFunctions(cte.Query)
				}
			}
		}

		// Validate columns
		for _, col := range s.Columns {
			p.validateExpressionFunctions(col)
		}

		// Validate FROM clause
		if s.TableExpr != nil {
			p.validateExpressionFunctions(s.TableExpr)
		}

		// Validate WHERE clause
		if s.Where != nil {
			p.validateExpressionFunctions(s.Where)
		}

		// Validate GROUP BY clause
		for _, expr := range s.GroupBy {
			p.validateExpressionFunctions(expr)
		}

		// Validate HAVING clause
		if s.Having != nil {
			p.validateExpressionFunctions(s.Having)
		}

		// Validate ORDER BY clause
		for _, order := range s.OrderBy {
			p.validateExpressionFunctions(order.Expression)
		}

		// Validate LIMIT and OFFSET clauses
		if s.Limit != nil {
			p.validateExpressionFunctions(s.Limit)
		}
		if s.Offset != nil {
			p.validateExpressionFunctions(s.Offset)
		}

	case *InsertStatement:
		// Validate VALUES expressions
		for _, row := range s.Values {
			for _, expr := range row {
				p.validateExpressionFunctions(expr)
			}
		}

	case *UpdateStatement:
		// Validate SET expressions
		for _, expr := range s.Updates {
			p.validateExpressionFunctions(expr)
		}

		// Validate WHERE clause
		if s.Where != nil {
			p.validateExpressionFunctions(s.Where)
		}

	case *DeleteStatement:
		// Validate WHERE clause
		if s.Where != nil {
			p.validateExpressionFunctions(s.Where)
		}

	case *MergeStatement:
		// Handle potentially incomplete MERGE statements
		if s == nil {
			return
		}

		// Check if fields are nil before validating
		if s.OnCondition != nil {
			// Validate ON condition
			p.validateExpressionFunctions(s.OnCondition)
		}

		// Validate source table if it's an expression (e.g., subquery)
		if s.SourceTable != nil {
			p.validateExpressionFunctions(s.SourceTable)
		}

		// Validate clauses if they exist
		if s.Clauses != nil {
			for _, clause := range s.Clauses {
				if clause == nil {
					continue
				}

				// Validate optional AND condition
				if clause.Condition != nil {
					p.validateExpressionFunctions(clause.Condition)
				}

				// Validate assignments map exists
				if clause.Assignments != nil && clause.Action == ActionUpdate {
					for _, expr := range clause.Assignments {
						if expr != nil {
							p.validateExpressionFunctions(expr)
						}
					}
				}

				// Validate INSERT values if they exist
				if clause.Values != nil && clause.Action == ActionInsert {
					for _, expr := range clause.Values {
						if expr != nil {
							p.validateExpressionFunctions(expr)
						}
					}
				}
			}
		}

	case *AlterTableStatement:
		// No function validation needed for ALTER TABLE
		// as it doesn't contain expressions with function calls

	case *CreateIndexStatement, *DropIndexStatement:
		// No function validation needed for index statements
		// as they don't contain expressions with function calls

	// Add transaction statement types
	case *BeginStatement, *CommitStatement, *RollbackStatement, *SavepointStatement:
		// No function validation needed for transaction statements
		// as they don't contain expressions with function calls

		// Add more statement types as needed
	}
}

// validateExpressionFunctions validates function calls in an expression
func (p *Parser) validateExpressionFunctions(expr Expression) {
	if expr == nil {
		return
	}

	switch e := expr.(type) {
	case *Parameter:
		// No validation needed for parameters
		// They'll be validated when bound during statement execution

	case *FunctionCall:
		// Validate the function call
		p.validateFunctionCall(e)

		// Recursively validate function arguments
		for _, arg := range e.Arguments {
			p.validateExpressionFunctions(arg)
		}

	case *InfixExpression:
		// Validate both sides of the expression
		p.validateExpressionFunctions(e.Left)
		p.validateExpressionFunctions(e.Right)

	case *PrefixExpression:
		// Validate the expression
		p.validateExpressionFunctions(e.Right)

	case *AliasedExpression:
		// Validate the expression
		p.validateExpressionFunctions(e.Expression)

	case *DistinctExpression:
		// Validate the underlying expression
		p.validateExpressionFunctions(e.Expr)

	case *SimpleTableSource:
		// No functions to validate

	case *SubqueryTableSource:
		// Validate the subquery
		if e.Subquery != nil {
			p.validateStatementFunctions(e.Subquery)
		}

	case *JoinTableSource:
		// Validate the left and right tables
		p.validateExpressionFunctions(e.Left)
		p.validateExpressionFunctions(e.Right)

		// Validate the join condition
		if e.Condition != nil {
			p.validateExpressionFunctions(e.Condition)
		}

	case *ScalarSubquery:
		// Validate the subquery
		if e.Subquery != nil {
			p.validateStatementFunctions(e.Subquery)
		}

	case *ExistsExpression:
		// Validate the subquery
		if e.Subquery != nil {
			p.validateStatementFunctions(e.Subquery)
		}

	case *InExpression:
		// Validate the left expression
		p.validateExpressionFunctions(e.Left)

		// Validate the right expression (could be a subquery or a list of values)
		p.validateExpressionFunctions(e.Right)

	case *ExpressionList:
		// Validate all expressions in the list
		for _, expr := range e.Expressions {
			p.validateExpressionFunctions(expr)
		}

	case *CaseExpression:
		// Validate the value expression if present
		if e.Value != nil {
			p.validateExpressionFunctions(e.Value)
		}

		// Validate all WHEN clauses
		for _, when := range e.WhenClauses {
			p.validateExpressionFunctions(when.Condition)
			p.validateExpressionFunctions(when.ThenResult)
		}

		// Validate the ELSE expression if present
		if e.ElseValue != nil {
			p.validateExpressionFunctions(e.ElseValue)
		}

	case *CastExpression:
		// Validate the expression being cast
		p.validateExpressionFunctions(e.Expr)

		// Add more expression types as needed
	}
}

// validateFunctionCall validates a function call using the function registry
func (p *Parser) validateFunctionCall(fc *FunctionCall) {
	// Special handling for star argument in COUNT(*) - it's always valid
	// and doesn't need type checking
	if len(fc.Arguments) == 1 && fc.Arguments[0].String() == "*" {
		// Lookup the function - but just to get its info
		funcInfo, err := p.funcRegistry.Get(fc.Function)
		if err != nil {
			p.addError(fmt.Sprintf("unknown function %s at %s", fc.Function, fc.Position()))
			return
		}

		// Ensure COUNT(*) is an aggregate function
		if funcInfo.Type != funcregistry.AggregateFunction {
			p.addError(fmt.Sprintf("star (*) argument only valid for aggregate functions at %s", fc.Position()))
			return
		}

		// Set the function info
		fc.FunctionInfo = &funcInfo
		return
	}

	// Determine argument types
	var argTypes []funcregistry.DataType

	for _, arg := range fc.Arguments {
		// This is a simple type inference; in a real system you would need
		// a more sophisticated type inference system
		argType := p.inferExpressionType(arg)
		argTypes = append(argTypes, argType)
	}

	// Validate the function call
	funcInfo := fc.FunctionInfo
	err := p.funcValidator.ValidateFunctionCall(fc.Function, &funcInfo, argTypes)

	if err != nil {
		// Add validation error
		p.addError(fmt.Sprintf("invalid function call %s: %v at %s",
			fc.Function, err, fc.Position()))
	} else {
		// Update function info in the AST
		fc.FunctionInfo = funcInfo

		// Special validation for DISTINCT
		if fc.IsDistinct && funcInfo.Type != funcregistry.AggregateFunction {
			p.addError(fmt.Sprintf("DISTINCT keyword only valid for aggregate functions at %s", fc.Position()))
		}
	}
}

// inferExpressionType infers the data type of an expression
func (p *Parser) inferExpressionType(expr Expression) funcregistry.DataType {
	if expr == nil {
		return funcregistry.TypeUnknown
	}

	switch e := expr.(type) {
	case *IntegerLiteral:
		return funcregistry.TypeInteger

	case *FloatLiteral:
		return funcregistry.TypeFloat

	case *StringLiteral:
		return funcregistry.TypeString

	case *BooleanLiteral:
		return funcregistry.TypeBoolean

	case *NullLiteral:
		return funcregistry.TypeUnknown

	case *Parameter:
		// Parameters can be of any type until bound
		return funcregistry.TypeAny

	case *Identifier:
		// Special case for * operator which is represented as an identifier in our AST
		if e.Value == "*" {
			return funcregistry.TypeAny
		}

		// For now, identifiers are assumed to be of type ANY
		// In a real system, you would need to look up the type from schema information
		return funcregistry.TypeAny

	case *QualifiedIdentifier:
		// Same for qualified identifiers
		return funcregistry.TypeAny

	case *DistinctExpression:
		// The type of DISTINCT expression is the same as the underlying expression
		return p.inferExpressionType(e.Expr)

	case *FunctionCall:
		// If the function info is available, return its return type
		if e.FunctionInfo != nil {
			return e.FunctionInfo.Signature.ReturnType
		}
		return funcregistry.TypeAny

	case *CaseExpression:
		// For a CASE expression, the type is determined by the THEN and ELSE branches
		// If all branches return the same type, use that; otherwise, use ANY

		// Start with the type of the first WHEN-THEN pair
		if len(e.WhenClauses) == 0 {
			return funcregistry.TypeUnknown
		}

		// Get the type of the first THEN result
		resultType := p.inferExpressionType(e.WhenClauses[0].ThenResult)

		// Check if all THEN results have the same type
		for i := 1; i < len(e.WhenClauses); i++ {
			thenType := p.inferExpressionType(e.WhenClauses[i].ThenResult)
			if thenType != resultType {
				// If different types, use ANY
				resultType = funcregistry.TypeAny
				break
			}
		}

		// Check the ELSE expression if present
		if e.ElseValue != nil {
			elseType := p.inferExpressionType(e.ElseValue)
			if elseType != resultType {
				// If ELSE has a different type, use ANY
				resultType = funcregistry.TypeAny
			}
		}

		return resultType

	case *CastExpression:
		// For a CAST expression, the type is determined by the target type
		switch strings.ToUpper(e.TypeName) {
		case "INTEGER", "INT":
			return funcregistry.TypeInteger
		case "FLOAT", "REAL", "DOUBLE":
			return funcregistry.TypeFloat
		case "STRING", "TEXT", "VARCHAR", "CHAR":
			return funcregistry.TypeString
		case "BOOLEAN", "BOOL":
			return funcregistry.TypeBoolean
		default:
			// For unknown types, return ANY
			return funcregistry.TypeAny
		}

	case *InfixExpression:
		// For simple operators, infer the type based on operands
		left := p.inferExpressionType(e.Left)
		right := p.inferExpressionType(e.Right)

		// If both types are the same, return that type
		if left == right {
			return left
		}

		// For math operations
		if e.Operator == "+" || e.Operator == "-" || e.Operator == "*" || e.Operator == "/" {
			// If either is a float, the result is a float
			if left == funcregistry.TypeFloat || right == funcregistry.TypeFloat {
				return funcregistry.TypeFloat
			}
			// Otherwise, it's an integer
			return funcregistry.TypeInteger
		}

		// For comparison operations
		if e.Operator == "=" || e.Operator == "<>" || e.Operator == "!=" ||
			e.Operator == "<" || e.Operator == ">" || e.Operator == "<=" || e.Operator == ">=" {
			return funcregistry.TypeBoolean
		}

		// For logical operations
		if e.Operator == "AND" || e.Operator == "OR" {
			return funcregistry.TypeBoolean
		}

		// For IN and NOT IN operations
		if e.Operator == "IN" || e.Operator == "NOT IN" {
			return funcregistry.TypeBoolean
		}

		return funcregistry.TypeAny

	case *PrefixExpression:
		// For prefix expressions like NOT or unary minus
		if e.Operator == "NOT" {
			return funcregistry.TypeBoolean
		}
		if e.Operator == "-" {
			rightType := p.inferExpressionType(e.Right)
			if rightType == funcregistry.TypeFloat {
				return funcregistry.TypeFloat
			}
			return funcregistry.TypeInteger
		}

		return funcregistry.TypeAny

	case *AliasedExpression:
		// The type of an aliased expression is the same as the underlying expression
		return p.inferExpressionType(e.Expression)

	case *ExistsExpression:
		// EXISTS always returns a boolean
		return funcregistry.TypeBoolean

	case *InExpression:
		// IN and NOT IN always return a boolean
		return funcregistry.TypeBoolean

	case *ScalarSubquery:
		// For now, assume ANY type for scalar subqueries
		// In a complete implementation, you would analyze the SELECT statement
		return funcregistry.TypeAny

	default:
		return funcregistry.TypeAny
	}
}

// GetParameterCount returns the number of parameters in the current statement
func (p *Parser) GetParameterCount() int {
	return p.parameterCounter
}

// parseStatement parses a SQL statement
func (p *Parser) parseStatement() Statement {
	// Reset parameter counter for each new statement and increment statement ID
	p.parameterStatementCounter = 0
	p.currentStatementID++

	// Default clause at the statement level
	p.currentClause = "UNKNOWN"

	// Check for WITH clause first - it can appear before any statement
	if p.curTokenIsKeyword("WITH") {
		withClause := p.parseWithClause()
		if withClause == nil {
			return nil
		}

		// The statement following the WITH clause
		// Currently only SELECT statements can have WITH clauses
		if p.curTokenIsKeyword("SELECT") {
			selectStmt := p.parseSelectStatement()
			if selectStmt == nil {
				return nil
			}
			selectStmt.With = withClause
			return selectStmt
		} else {
			p.addError(fmt.Sprintf("expected SELECT after WITH clause at %s", p.curToken.Position))
			return nil
		}
	}

	switch {
	case p.curTokenIsKeyword("SELECT"):
		return p.parseSelectStatement()
	case p.curTokenIsKeyword("CREATE"):
		return p.parseCreateStatement()
	case p.curTokenIsKeyword("DROP"):
		return p.parseDropStatement()
	case p.curTokenIsKeyword("INSERT"):
		p.currentClause = "INSERT VALUES"
		return p.parseInsertStatement()
	case p.curTokenIsKeyword("UPDATE"):
		p.currentClause = "UPDATE SET"
		return p.parseUpdateStatement()
	case p.curTokenIsKeyword("DELETE"):
		p.currentClause = "WHERE"
		return p.parseDeleteStatement()
	case p.curTokenIsKeyword("ALTER"):
		return p.parseAlterStatement()
	case p.curTokenIsKeyword("MERGE"):
		return p.parseMergeStatement()
	case p.curTokenIsKeyword("BEGIN"):
		return p.parseBeginStatement()
	case p.curTokenIsKeyword("COMMIT"):
		return p.parseCommitStatement()
	case p.curTokenIsKeyword("ROLLBACK"):
		return p.parseRollbackStatement()
	case p.curTokenIsKeyword("SAVEPOINT"):
		return p.parseSavepointStatement()
	case p.curTokenIsKeyword("SET"):
		return p.parseSetStatement()
	case p.curTokenIsKeyword("PRAGMA"):
		return p.parsePragmaStatement()
	case p.curTokenIsKeyword("SHOW"):
		return p.parseShowStatement()
	case p.curTokenIs(TokenEOF):
		return nil // Just skip EOF tokens
	default:
		// Try to parse as an expression statement (useful for CAST expressions, etc.)
		expr := p.parseExpression(LOWEST)
		if expr != nil {
			stmt := &ExpressionStatement{
				Token:      p.curToken,
				Expression: expr,
			}
			return stmt
		}

		p.addError(fmt.Sprintf("unexpected token: %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}
}

// parseWithClause parses a WITH clause
func (p *Parser) parseWithClause() *WithClause {
	// Create a WITH clause with the current token
	withClause := &WithClause{
		Token: p.curToken,
		CTEs:  []*CommonTableExpression{},
	}

	// Check for RECURSIVE keyword
	if p.peekTokenIsKeyword("RECURSIVE") {
		p.nextToken() // consume RECURSIVE
		withClause.IsRecursive = true
	}

	// Parse the first CTE
	p.nextToken() // Move to first CTE name
	cte := p.parseCommonTableExpression()
	if cte == nil {
		return nil
	}

	// Set recursive flag on the CTE based on the WITH clause
	cte.IsRecursive = withClause.IsRecursive
	withClause.CTEs = append(withClause.CTEs, cte)

	// Parse additional CTEs
	for p.peekTokenIsPunctuator(",") {
		p.nextToken() // consume comma
		p.nextToken() // Move to next CTE name

		cte = p.parseCommonTableExpression()
		if cte == nil {
			return nil
		}

		// Set recursive flag on the CTE
		cte.IsRecursive = withClause.IsRecursive
		withClause.CTEs = append(withClause.CTEs, cte)
	}

	// Advance to the next token after the WITH clause
	p.nextToken()

	return withClause
}

// parseCommonTableExpression parses a single CTE definition
func (p *Parser) parseCommonTableExpression() *CommonTableExpression {
	// CTE must start with an identifier (name)
	if !p.curTokenIs(TokenIdentifier) {
		p.addError(fmt.Sprintf("expected identifier as CTE name, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Create a CTE with the current token
	cte := &CommonTableExpression{
		Token: p.curToken,
		Name: &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		},
		ColumnNames: []*Identifier{},
	}

	// Check for optional column list
	if p.peekTokenIsPunctuator("(") {
		p.nextToken() // consume (

		// Parse column names
		for {
			p.nextToken() // Move to column name

			if !p.curTokenIs(TokenIdentifier) {
				p.addError(fmt.Sprintf("expected identifier as column name, got %s at %s", p.curToken.Literal, p.curToken.Position))
				return nil
			}

			colName := &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			}

			cte.ColumnNames = append(cte.ColumnNames, colName)

			// Check for comma or closing parenthesis
			if p.peekTokenIsPunctuator(")") {
				p.nextToken() // consume )
				break
			} else if p.peekTokenIsPunctuator(",") {
				p.nextToken() // consume comma
			} else {
				p.addError(fmt.Sprintf("expected ',' or ')' in column list, got %s at %s", p.peekToken.Literal, p.peekToken.Position))
				return nil
			}
		}
	}

	// Expect AS
	if !p.expectKeyword("AS") {
		return nil
	}

	// Expect opening parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
		p.addError(fmt.Sprintf("expected '(' after AS, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse the query (must be a SELECT statement)
	p.nextToken() // Move to SELECT

	if !p.curTokenIsKeyword("SELECT") {
		p.addError(fmt.Sprintf("expected SELECT in CTE definition, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	cte.Query = p.parseSelectStatement()
	if cte.Query == nil {
		return nil
	}

	// Expect closing parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
		p.addError(fmt.Sprintf("expected ')' after CTE query, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	return cte
}

// parseSelectStatement parses a SELECT statement
func (p *Parser) parseSelectStatement() *SelectStatement {
	// Create a SELECT statement with the current token
	stmt := &SelectStatement{
		Token: p.curToken,
	}

	// Set clause to SELECT for the column list
	p.currentClause = "SELECT"

	// Check for DISTINCT
	if p.peekTokenIsKeyword("DISTINCT") {
		p.nextToken()
		stmt.Distinct = true
	}

	// Parse the column list
	stmt.Columns = p.parseExpressionList()

	// Parse the FROM clause
	if p.peekTokenIsKeyword("FROM") {
		p.nextToken() // consume FROM
		p.currentClause = "FROM"
		stmt.TableExpr = p.parseTableExpression()
	}

	// Parse the WHERE clause
	if p.peekTokenIsKeyword("WHERE") {
		p.nextToken() // consume WHERE
		p.nextToken() // move to the expression
		p.currentClause = "WHERE"

		// Special handling for NOT at the beginning of WHERE clause
		if p.curTokenIsKeyword("NOT") {
			notToken := p.curToken
			p.nextToken() // Move past NOT

			// Parse the rest of the expression
			expr := p.parseExpression(LOWEST)

			// Create a NOT prefix expression
			stmt.Where = &PrefixExpression{
				Token:    notToken,
				Operator: "NOT",
				Right:    expr,
			}
		} else {
			stmt.Where = p.parseExpression(LOWEST)
		}
	}

	// Parse the GROUP BY clause
	if p.peekTokenIsKeyword("GROUP") {
		p.nextToken() // consume GROUP
		if !p.expectKeyword("BY") {
			return nil
		}
		p.currentClause = "GROUP BY"
		stmt.GroupBy = p.parseExpressionList()
	}

	// Parse the HAVING clause
	if p.peekTokenIsKeyword("HAVING") {
		p.nextToken() // consume HAVING
		p.nextToken() // advance to the expression
		p.currentClause = "HAVING"
		stmt.Having = p.parseExpression(LOWEST)
	}

	// Parse the ORDER BY clause
	if p.peekTokenIsKeyword("ORDER") {
		p.nextToken() // consume ORDER
		if !p.expectKeyword("BY") {
			return nil
		}
		p.currentClause = "ORDER BY"
		stmt.OrderBy = p.parseOrderByExpressions()
	}

	// Parse the LIMIT clause
	if p.peekTokenIsKeyword("LIMIT") {
		p.nextToken() // consume LIMIT
		p.nextToken() // advance to the limit value
		p.currentClause = "LIMIT"
		stmt.Limit = p.parseExpression(LOWEST)
	}

	// Parse the OFFSET clause
	if p.peekTokenIsKeyword("OFFSET") {
		p.nextToken() // consume OFFSET
		p.nextToken() // advance to the offset value
		p.currentClause = "OFFSET"
		stmt.Offset = p.parseExpression(LOWEST)
	}

	return stmt
}

// parseExpressionList parses a comma-separated list of expressions
func (p *Parser) parseExpressionList() []Expression {
	list := []Expression{}

	// If just a star, return it - now * is an operator, not a punctuator
	if p.peekTokenIsOperator("*") {
		p.nextToken()
		list = append(list, &Identifier{
			Token: p.curToken,
			Value: "*",
		})
		return list
	}

	p.nextToken() // Move to the first expression
	expr := p.parseExpression(LOWEST)

	// Special handling for column aliases in SELECT
	if p.peekTokenIsKeyword("AS") {
		p.nextToken() // consume AS
		// Expect identifier for the alias
		if p.expectPeek(TokenIdentifier) {
			expr = &AliasedExpression{
				Token:      Token{Type: TokenKeyword, Literal: "AS", Position: p.curToken.Position},
				Expression: expr,
				Alias: &Identifier{
					Token: p.curToken,
					Value: p.curToken.Literal,
				},
			}
		}
	}

	list = append(list, expr)

	for p.peekTokenIsPunctuator(",") {
		p.nextToken() // consume comma
		p.nextToken() // Move to the next expression
		expr := p.parseExpression(LOWEST)

		// Special handling for column aliases in SELECT
		if p.peekTokenIsKeyword("AS") {
			p.nextToken() // consume AS
			// Expect identifier for the alias
			if p.expectPeek(TokenIdentifier) {
				expr = &AliasedExpression{
					Token:      Token{Type: TokenKeyword, Literal: "AS", Position: p.curToken.Position},
					Expression: expr,
					Alias: &Identifier{
						Token: p.curToken,
						Value: p.curToken.Literal,
					},
				}
			}
		}

		list = append(list, expr)
	}

	return list
}

// parseOrderByExpression parses a single ORDER BY expression
func (p *Parser) parseOrderByExpression() OrderByExpression {
	// Parse the expression for ORDER BY
	// This can be a column name, an aliased column, or another expression
	var expr Expression

	// Check if we're looking at a keyword that might be a column name (like TIMESTAMP)
	if p.curTokenIs(TokenKeyword) {
		// Treat keywords as identifiers when they're column names in ORDER BY context
		expr = &Identifier{
			Token: p.curToken,
			Value: strings.ToLower(p.curToken.Literal), // Use lowercase for column names
		}
		p.nextToken() // Move past this identifier
	} else {
		expr = p.parseExpression(LOWEST)
		if expr == nil {
			// If we failed to parse an expression, use a placeholder identifier
			expr = &Identifier{
				Token: p.curToken,
				Value: "error_placeholder",
			}
		}
	}

	orderExpr := OrderByExpression{
		Expression: expr,
		Ascending:  true, // Default to ASC
	}

	// Check for ASC/DESC
	if p.peekTokenIsKeyword("ASC") {
		p.nextToken()
		orderExpr.Ascending = true
	} else if p.peekTokenIsKeyword("DESC") {
		p.nextToken()
		orderExpr.Ascending = false
	}

	return orderExpr
}

// parseOrderByExpressions parses ORDER BY expressions
func (p *Parser) parseOrderByExpressions() []OrderByExpression {
	list := []OrderByExpression{}

	p.nextToken() // Move to the first expression

	// Parse the first ORDER BY expression
	orderExpr := p.parseOrderByExpression()
	list = append(list, orderExpr)

	// Parse additional ORDER BY expressions
	for p.peekTokenIsPunctuator(",") {
		p.nextToken() // consume comma
		p.nextToken() // Move to the next expression

		orderExpr := p.parseOrderByExpression()
		list = append(list, orderExpr)
	}

	return list
}

// JoinType represents a type of join
type JoinType int

const (
	// InnerJoin is an inner join
	InnerJoin JoinType = iota
	// LeftJoin is a left outer join
	LeftJoin
	// RightJoin is a right outer join
	RightJoin
	// FullJoin is a full outer join
	FullJoin
	// CrossJoin is a cross join
	CrossJoin
)

// JoinType constants for parsing join clauses
const (
	INNER_JOIN = "INNER"
	LEFT_JOIN  = "LEFT"
	RIGHT_JOIN = "RIGHT"
	FULL_JOIN  = "FULL"
	CROSS_JOIN = "CROSS"
)

// ParseJoinType converts a join type string to a JoinType enum
func ParseJoinType(joinTypeStr string) JoinType {
	switch joinTypeStr {
	case INNER_JOIN:
		return InnerJoin
	case LEFT_JOIN:
		return LeftJoin
	case RIGHT_JOIN:
		return RightJoin
	case FULL_JOIN:
		return FullJoin
	case CROSS_JOIN:
		return CrossJoin
	default:
		return InnerJoin
	}
}

// AggregateType represents a type of aggregate function
type AggregateType int

const (
	// Count is the COUNT aggregate function
	Count AggregateType = iota
	// Sum is the SUM aggregate function
	Sum
	// Avg is the AVG aggregate function
	Avg
	// Min is the MIN aggregate function
	Min
	// Max is the MAX aggregate function
	Max
)

// AggregateFunction represents an aggregate function
type AggregateFunction struct {
	Type       AggregateType
	Column     string
	Alias      string
	IsDistinct bool
}

// UpdateStatement represents an UPDATE statement
type UpdateStatement struct {
	Token     Token // The UPDATE token
	TableName *Identifier
	Updates   map[string]Expression
	Where     Expression
}

// DeleteStatement represents a DELETE statement
type DeleteStatement struct {
	Token     Token // The DELETE token
	TableName *Identifier
	Where     Expression
}

func (us *UpdateStatement) statementNode()       {}
func (us *UpdateStatement) TokenLiteral() string { return us.Token.Literal }
func (us *UpdateStatement) Position() Position   { return us.Token.Position }
func (us *UpdateStatement) String() string {
	var result string
	result = "UPDATE " + us.TableName.String() + " SET "

	i := 0
	for col, val := range us.Updates {
		if i > 0 {
			result += ", "
		}
		result += col + " = " + val.String()
		i++
	}

	if us.Where != nil {
		result += " WHERE " + us.Where.String()
	}

	return result
}

func (ds *DeleteStatement) statementNode()       {}
func (ds *DeleteStatement) TokenLiteral() string { return ds.Token.Literal }
func (ds *DeleteStatement) Position() Position   { return ds.Token.Position }
func (ds *DeleteStatement) String() string {
	var result string
	result = "DELETE FROM " + ds.TableName.String()

	if ds.Where != nil {
		result += " WHERE " + ds.Where.String()
	}

	return result
}

// parseTableExpression parses a table expression in the FROM clause
func (p *Parser) parseTableExpression() Expression {
	p.nextToken() // Move to the table name or subquery
	expr := p.parseSimpleTableExpression()

	// Look for JOIN clauses
	for p.peekTokenIsKeyword("JOIN") || p.peekTokenIsKeyword("INNER") ||
		p.peekTokenIsKeyword("LEFT") || p.peekTokenIsKeyword("RIGHT") ||
		p.peekTokenIsKeyword("FULL") || p.peekTokenIsKeyword("CROSS") {

		expr = p.parseJoinTableExpression(expr)
		if expr == nil {
			return nil
		}
	}

	return expr
}

// isCTEName checks if the given name matches a CTE defined in the current statement
// This is a utility method for CTE reference resolution
func (p *Parser) isCTEName(name string) bool {
	_ = name // Placeholder for actual implementation

	// This is a simple implementation that assumes we're parsing a SQL statement
	// that might have WITH clause. In a real implementation, you'd need to track
	// the CTEs defined in the current context.

	// At this point in parsing, we can check if we've seen a WITH clause and
	// if the identifier matches one of the CTE names.
	// For simplicity, we'll check if the current SELECT statement being parsed
	// has a WITH clause, and if the name matches any of the CTEs.

	// This is a placeholder implementation - in a real parser, you would
	// maintain a symbol table or scope stack to track CTE names.
	// For now, just return false and assume no CTEs.
	return false
}

// parseSimpleTableExpression parses a simple table reference
func (p *Parser) parseSimpleTableExpression() Expression {
	var tableExpr Expression

	// Handle table name or CTE reference
	if p.curTokenIs(TokenIdentifier) {
		identifier := &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		}

		// Check if this identifier is a CTE name
		// This would be done by checking the current context, but for now
		// we'll just check if we're in a statement with a WITH clause
		if p.isCTEName(identifier.Value) {
			// This is a CTE reference
			tableExpr = &CTEReference{
				Token: p.curToken,
				Name:  identifier,
			}

			// Check for AS clause
			if p.peekTokenIsKeyword("AS") {
				p.nextToken() // consume AS
				if !p.expectPeek(TokenIdentifier) {
					return nil
				}

				// Set alias
				cteRef := tableExpr.(*CTEReference)
				cteRef.Alias = &Identifier{
					Token: p.curToken,
					Value: p.curToken.Literal,
				}
			} else if p.peekTokenIs(TokenIdentifier) {
				// Implicit alias (e.g., "FROM cte_name alias")
				p.nextToken() // consume the alias

				// Set alias
				cteRef := tableExpr.(*CTEReference)
				cteRef.Alias = &Identifier{
					Token: p.curToken,
					Value: p.curToken.Literal,
				}
			}
		} else {
			// This is a regular table name
			tableExpr = &SimpleTableSource{
				Token: p.curToken,
				Name:  identifier,
			}

			// Check for AS clause
			if p.peekTokenIsKeyword("AS") {
				p.nextToken() // consume AS

				// Check if this is AS OF
				if p.peekTokenIsKeyword("OF") {
					p.nextToken() // consume OF

					// Parse AS OF type (TRANSACTION or TIMESTAMP)
					if !p.expectPeek(TokenKeyword) {
						p.addError(fmt.Sprintf("expected TRANSACTION or TIMESTAMP after AS OF at %s", p.curToken.Position))
						return nil
					}

					asOfType := strings.ToUpper(p.curToken.Literal)
					if asOfType != "TRANSACTION" && asOfType != "TIMESTAMP" {
						p.addError(fmt.Sprintf("expected TRANSACTION or TIMESTAMP after AS OF, got %s at %s", asOfType, p.curToken.Position))
						return nil
					}

					// Parse the value
					p.nextToken()
					value := p.parseExpression(LOWEST)
					if value == nil {
						return nil
					}

					// Set AS OF clause
					simpleTable := tableExpr.(*SimpleTableSource)
					simpleTable.AsOf = &AsOfClause{
						Token: p.curToken,
						Type:  asOfType,
						Value: value,
					}

					// Check for alias after AS OF
					if p.peekTokenIsKeyword("AS") {
						p.nextToken() // consume AS
						if !p.expectPeek(TokenIdentifier) {
							return nil
						}
						simpleTable.Alias = &Identifier{
							Token: p.curToken,
							Value: p.curToken.Literal,
						}
					} else if p.peekTokenIs(TokenIdentifier) {
						p.nextToken() // consume the alias
						simpleTable.Alias = &Identifier{
							Token: p.curToken,
							Value: p.curToken.Literal,
						}
					}
				} else {
					// Regular AS alias
					if !p.expectPeek(TokenIdentifier) {
						return nil
					}

					// Set alias
					simpleTable := tableExpr.(*SimpleTableSource)
					simpleTable.Alias = &Identifier{
						Token: p.curToken,
						Value: p.curToken.Literal,
					}
				}
			} else if p.peekTokenIs(TokenIdentifier) {
				// Implicit alias (e.g., "FROM users u")
				p.nextToken() // consume the alias

				// Set alias
				simpleTable := tableExpr.(*SimpleTableSource)
				simpleTable.Alias = &Identifier{
					Token: p.curToken,
					Value: p.curToken.Literal,
				}
			}
		}

		return tableExpr
	} else if p.curTokenIsPunctuator("(") {
		// Save the token for the subquery
		token := p.curToken

		// Parse subquery
		p.nextToken() // Move past the opening parenthesis

		// Check if this is a SELECT statement
		if !p.curTokenIsKeyword("SELECT") {
			p.addError(fmt.Sprintf("expected SELECT in subquery, got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		// Parse the SELECT statement
		selectStmt := p.parseSelectStatement()
		if selectStmt == nil {
			return nil
		}

		// Expect closing parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected closing parenthesis, got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		// Create the subquery table source
		subquery := &SubqueryTableSource{
			Token:    token,
			Subquery: selectStmt,
		}

		// Check for alias (required for subqueries in FROM)
		if p.peekTokenIsKeyword("AS") {
			p.nextToken() // consume AS
			if !p.expectPeek(TokenIdentifier) {
				return nil
			}

			subquery.Alias = &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			}
		} else if p.peekTokenIs(TokenIdentifier) {
			// Implicit alias
			p.nextToken() // consume the alias

			subquery.Alias = &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			}
		} else {
			p.addError(fmt.Sprintf("subquery in FROM must have an alias at %s", p.curToken.Position))
			return nil
		}

		return subquery
	}

	p.addError(fmt.Sprintf("expected table name or subquery, got %s at %s", p.curToken.Literal, p.curToken.Position))
	return nil
}

// parseJoinTableExpression parses a JOIN clause
func (p *Parser) parseJoinTableExpression(left Expression) Expression {
	// Ensure left expression is a TableSource
	leftTable, ok := left.(TableSource)
	if !ok {
		p.addError(fmt.Sprintf("expected table source on left side of JOIN, got %T at %s", left, p.curToken.Position))
		return nil
	}

	// Determine join type
	joinType := INNER_JOIN // Default join type
	joinToken := p.peekToken

	if p.peekTokenIsKeyword("INNER") || p.peekTokenIsKeyword("LEFT") ||
		p.peekTokenIsKeyword("RIGHT") || p.peekTokenIsKeyword("FULL") ||
		p.peekTokenIsKeyword("CROSS") {
		p.nextToken() // consume join type
		joinType = p.curToken.Literal

		// For OUTER joins
		if p.peekTokenIsKeyword("OUTER") {
			p.nextToken() // consume OUTER
		}

		if !p.expectKeyword("JOIN") {
			return nil
		}
	} else {
		// Plain JOIN keyword (implicit INNER JOIN)
		p.nextToken() // consume JOIN
	}

	// Parse the right table expression
	p.nextToken() // Move to the right table name
	rightExpr := p.parseSimpleTableExpression()
	if rightExpr == nil {
		return nil
	}

	rightTable, ok := rightExpr.(TableSource)
	if !ok {
		p.addError(fmt.Sprintf("expected table source on right side of JOIN, got %T", rightExpr))
		return nil
	}

	// Create the join table source
	joinExpr := &JoinTableSource{
		Token:    joinToken,
		Left:     leftTable,
		JoinType: joinType,
		Right:    rightTable,
	}

	// Parse the join condition (ON clause)
	if p.peekTokenIsKeyword("ON") {
		p.nextToken() // consume ON
		p.nextToken() // move to condition expression
		condition := p.parseExpression(LOWEST)
		if condition == nil {
			return nil
		}
		joinExpr.Condition = condition
	} else if p.peekTokenIsKeyword("USING") {
		// USING clause not implemented yet
		p.addError(fmt.Sprintf("USING clause not yet supported at %s", p.curToken.Position))
		return nil
	} else if joinType != CROSS_JOIN {
		// JOIN requires ON or USING clause, except for CROSS JOIN
		p.addError(fmt.Sprintf("expected ON or USING clause after JOIN at %s", p.curToken.Position))
		return nil
	}

	return joinExpr
}

// parseCreateStatement parses a CREATE statement
func (p *Parser) parseCreateStatement() Statement {
	// Create a CREATE statement with the current token
	token := p.curToken

	// Check what we're creating
	if p.peekTokenIsKeyword("TABLE") {
		p.nextToken() // consume TABLE
		return p.parseCreateTableStatement(token)
	} else if p.peekTokenIsKeyword("UNIQUE") {
		p.nextToken() // consume UNIQUE

		// Check if this is a UNIQUE COLUMNAR INDEX
		if p.peekTokenIsKeyword("COLUMNAR") {
			p.nextToken() // consume COLUMNAR
			if p.peekTokenIsKeyword("INDEX") {
				p.nextToken() // consume INDEX
				stmt := p.parseCreateColumnarIndexStatement(token)
				if stmt != nil {
					stmt.(*CreateColumnarIndexStatement).IsUnique = true
				}
				return stmt
			}
			p.addError(fmt.Sprintf("expected INDEX after COLUMNAR at %s", p.curToken.Position))
			return nil
		}

		// If it's not COLUMNAR, it must be a regular UNIQUE INDEX
		// Since we've already consumed UNIQUE, we can't use parseCreateIndexStatement
		// So we'll manually parse a CREATE UNIQUE INDEX statement
		stmt := &CreateIndexStatement{
			Token:    token,
			IsUnique: true,
		}

		// We've already consumed UNIQUE, now expect INDEX
		if !p.expectKeyword("INDEX") {
			return nil
		}

		// Continue parsing the rest of the index statement
		// Check for IF NOT EXISTS
		if p.peekTokenIsKeyword("IF") {
			p.nextToken() // consume IF
			if !p.expectKeyword("NOT") {
				return nil
			}
			if !p.expectKeyword("EXISTS") {
				return nil
			}
			stmt.IfNotExists = true
		}

		// Parse the index name
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}
		stmt.IndexName = &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		}

		// Expect ON keyword
		if !p.expectKeyword("ON") {
			return nil
		}

		// Parse the table name
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}
		stmt.TableName = &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		}

		// Parse the column list
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
			p.addError(fmt.Sprintf("expected '(', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		// Parse columns
		stmt.Columns = p.parseIdentifierList()

		// Expect closing parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		return stmt
	} else if p.peekTokenIsKeyword("INDEX") {
		return p.parseCreateIndexStatement(token)
	} else if p.peekTokenIsKeyword("COLUMNAR") {
		p.nextToken() // consume COLUMNAR
		if p.peekTokenIsKeyword("INDEX") {
			p.nextToken() // consume INDEX
			return p.parseCreateColumnarIndexStatement(token)
		}
		p.addError(fmt.Sprintf("expected INDEX after COLUMNAR at %s", p.curToken.Position))
		return nil
	} else if p.peekTokenIsKeyword("VIEW") {
		p.nextToken() // consume VIEW
		return p.parseCreateViewStatement()
	}

	p.addError(fmt.Sprintf("expected TABLE, INDEX, COLUMNAR INDEX, or VIEW after CREATE at %s", p.curToken.Position))
	return nil
}

// parseCreateTableStatement parses a CREATE TABLE statement
func (p *Parser) parseCreateTableStatement(token Token) Statement {
	stmt := &CreateTableStatement{
		Token: token,
	}

	// Check for IF NOT EXISTS
	if p.peekTokenIsKeyword("IF") {
		p.nextToken() // consume IF
		if !p.expectKeyword("NOT") {
			return nil
		}
		if !p.expectKeyword("EXISTS") {
			return nil
		}
		stmt.IfNotExists = true
	}

	// Parse the table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Expect opening parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
		p.addError(fmt.Sprintf("expected '(', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse column definitions
	stmt.Columns = p.parseColumnDefinitions()

	// Expect closing parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
		p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	return stmt
}

// parseCreateIndexStatement parses a CREATE INDEX statement
func (p *Parser) parseCreateIndexStatement(token Token) Statement {
	stmt := &CreateIndexStatement{
		Token: token,
	}

	// Check for UNIQUE
	if p.peekTokenIsKeyword("UNIQUE") {
		p.nextToken() // consume UNIQUE
		stmt.IsUnique = true

		// Now expect INDEX
		if !p.expectKeyword("INDEX") {
			return nil
		}
	} else {
		// Not UNIQUE, so expect INDEX
		if !p.expectKeyword("INDEX") {
			return nil
		}
	}

	// Check for IF NOT EXISTS
	if p.peekTokenIsKeyword("IF") {
		p.nextToken() // consume IF
		if !p.expectKeyword("NOT") {
			return nil
		}
		if !p.expectKeyword("EXISTS") {
			return nil
		}
		stmt.IfNotExists = true
	}

	// Parse the index name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.IndexName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Expect ON keyword
	if !p.expectKeyword("ON") {
		return nil
	}

	// Parse the table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Expect opening parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
		p.addError(fmt.Sprintf("expected '(', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse column list
	stmt.Columns = p.parseIdentifierList()

	// Expect closing parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
		p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	return stmt
}

// parseColumnDefinitions parses column definitions in a CREATE TABLE statement
func (p *Parser) parseColumnDefinitions() []ColumnDefinition {
	columns := []ColumnDefinition{}

	// Parse first column
	p.nextToken()
	col := p.parseColumnDefinition()
	columns = append(columns, col)

	// Parse additional columns
	for p.peekTokenIsPunctuator(",") {
		p.nextToken() // consume comma
		p.nextToken() // move to column name
		col = p.parseColumnDefinition()
		columns = append(columns, col)
	}

	return columns
}

// parseColumnDefinition parses a single column definition
func (p *Parser) parseColumnDefinition() ColumnDefinition {
	col := ColumnDefinition{}

	// Parse column name
	if !p.curTokenIs(TokenIdentifier) {
		p.addError(fmt.Sprintf("expected column name, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return col
	}

	col.Name = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Parse column type
	if !p.expectPeek(TokenKeyword) {
		return col
	}
	col.Type = p.curToken.Literal

	// Parse column constraints
	for p.peekTokenIs(TokenKeyword) {
		p.nextToken()

		switch strings.ToUpper(p.curToken.Literal) {
		case "PRIMARY":
			// PRIMARY KEY constraint
			primaryToken := p.curToken
			if !p.expectKeyword("KEY") {
				return col
			}
			// Create and store a PRIMARY KEY constraint
			pkConstraint := &PrimaryKeyConstraint{
				Token: primaryToken,
			}
			col.Constraints = append(col.Constraints, pkConstraint)

		case "NOT":
			// NOT NULL constraint
			notToken := p.curToken
			if !p.expectKeyword("NULL") {
				return col
			}
			// Create and store a NOT NULL constraint
			nnConstraint := &NotNullConstraint{
				Token: notToken,
			}
			col.Constraints = append(col.Constraints, nnConstraint)

		case "DEFAULT":
			// DEFAULT value constraint
			p.nextToken() // move to the default value
			// The default value could be a literal (string, number, boolean, etc.)
			// For now, we just skip over it
			if p.curTokenIs(TokenKeyword) &&
				(p.curToken.Literal == "TRUE" || p.curToken.Literal == "FALSE" || p.curToken.Literal == "NULL") {
				// It's a keyword literal, already consumed
			} else if p.curTokenIs(TokenString) || p.curTokenIs(TokenInteger) || p.curTokenIs(TokenFloat) {
				// It's a literal, already consumed
			} else {
				// Unexpected token for default value
				p.addError(fmt.Sprintf("unexpected token: for DEFAULT value: %s at %s", p.curToken.Literal, p.curToken.Position))
			}

		case "UNIQUE", "CHECK", "REFERENCES", "COLLATE":
			// Other constraints
			// For now, we just recognize the keyword but don't parse further

		default:
			// Unknown constraint or not a constraint at all
			p.nextToken() // go back to previous token
			return col
		}
	}

	return col
}

// parseDropStatement parses a DROP statement
func (p *Parser) parseDropStatement() Statement {
	// Create a DROP statement with the current token
	token := p.curToken

	// Check what we're dropping
	if p.peekTokenIsKeyword("TABLE") {
		p.nextToken() // consume TABLE
		return p.parseDropTableStatement(token)
	} else if p.peekTokenIsKeyword("INDEX") {
		p.nextToken() // consume INDEX
		return p.parseDropIndexStatement(token)
	} else if p.peekTokenIsKeyword("COLUMNAR") {
		p.nextToken() // consume COLUMNAR
		if p.peekTokenIsKeyword("INDEX") {
			p.nextToken() // consume INDEX
			return p.parseDropColumnarIndexStatement(token)
		}
		p.addError(fmt.Sprintf("expected INDEX after COLUMNAR at %s", p.curToken.Position))
		return nil
	} else if p.peekTokenIsKeyword("VIEW") {
		p.nextToken() // consume VIEW
		return p.parseDropViewStatement()
	}

	p.addError(fmt.Sprintf("expected TABLE, INDEX, COLUMNAR INDEX, or VIEW after DROP at %s", p.curToken.Position))
	return nil
}

// parseCreateColumnarIndexStatement parses a CREATE COLUMNAR INDEX statement
func (p *Parser) parseCreateColumnarIndexStatement(token Token) Statement {
	stmt := &CreateColumnarIndexStatement{
		Token: token,
	}

	// Check for IF NOT EXISTS
	if p.peekTokenIsKeyword("IF") {
		p.nextToken() // consume IF
		if !p.expectKeyword("NOT") {
			return nil
		}
		if !p.expectKeyword("EXISTS") {
			return nil
		}
		stmt.IfNotExists = true
	}

	// Expect ON keyword for the table name
	if !p.expectKeyword("ON") {
		return nil
	}

	// Parse the table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Expect opening parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
		p.addError(fmt.Sprintf("expected '(', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse the column name (only one column for columnar indexes)
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.ColumnName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Expect closing parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
		p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	return stmt
}

// parseDropColumnarIndexStatement parses a DROP COLUMNAR INDEX statement
func (p *Parser) parseDropColumnarIndexStatement(token Token) Statement {
	stmt := &DropColumnarIndexStatement{
		Token: token,
	}

	// Check for IF EXISTS
	if p.peekTokenIsKeyword("IF") {
		p.nextToken() // consume IF
		if !p.expectKeyword("EXISTS") {
			return nil
		}
		stmt.IfExists = true
	}

	// Expect ON keyword for the table name
	if !p.expectKeyword("ON") {
		return nil
	}

	// Parse the table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Expect opening parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
		p.addError(fmt.Sprintf("expected '(', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse the column name (only one column for columnar indexes)
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.ColumnName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Expect closing parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
		p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	return stmt
}

// parseDropTableStatement parses a DROP TABLE statement
func (p *Parser) parseDropTableStatement(token Token) Statement {
	stmt := &DropTableStatement{
		Token: token,
	}

	// Check for IF EXISTS
	if p.peekTokenIsKeyword("IF") {
		p.nextToken() // consume IF
		if !p.expectKeyword("EXISTS") {
			return nil
		}
		stmt.IfExists = true
	}

	// Parse the table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	return stmt
}

// parseDropIndexStatement parses a DROP INDEX statement
func (p *Parser) parseDropIndexStatement(token Token) Statement {
	stmt := &DropIndexStatement{
		Token: token,
	}

	// Check for IF EXISTS
	if p.peekTokenIsKeyword("IF") {
		p.nextToken() // consume IF
		if !p.expectKeyword("EXISTS") {
			return nil
		}
		stmt.IfExists = true
	}

	// Parse the index name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.IndexName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Check for optional ON clause
	if p.peekTokenIsKeyword("ON") {
		p.nextToken() // consume ON

		// Parse the table name
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}
		stmt.TableName = &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		}
	}

	return stmt
}

// parseMergeStatement parses a MERGE statement
func (p *Parser) parseMergeStatement() Statement {
	// Create a MERGE statement with the current token
	stmt := &MergeStatement{
		Token:   p.curToken,
		Clauses: []*MergeClause{},
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file after MERGE at %s", p.curToken.Position))
		return nil
	}

	// Expect INTO
	if !p.expectKeyword("INTO") {
		return nil
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file after INTO at %s", p.curToken.Position))
		return nil
	}

	// Parse target table name
	if !p.expectPeek(TokenIdentifier) {
		p.addError(fmt.Sprintf("expected table name after INTO, got %s at %s",
			p.peekToken.Literal, p.peekToken.Position))
		return nil
	}
	stmt.TargetTable = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Check for AS keyword and target alias
	if p.peekToken.Type != TokenEOF && p.peekTokenIsKeyword("AS") {
		p.nextToken() // consume AS
		if !p.expectPeek(TokenIdentifier) {
			p.addError(fmt.Sprintf("expected table alias after AS, got %s at %s",
				p.peekToken.Literal, p.peekToken.Position))
			return nil
		}
		stmt.TargetAlias = &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		}
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file, expected USING at %s", p.curToken.Position))
		return nil
	}

	// Expect USING
	if !p.expectKeyword("USING") {
		return nil
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file after USING at %s", p.curToken.Position))
		return nil
	}

	// Parse source table (currently only supports table names, not subqueries)
	if !p.expectPeek(TokenIdentifier) {
		p.addError(fmt.Sprintf("expected source table name after USING, got %s at %s",
			p.peekToken.Literal, p.peekToken.Position))
		return nil
	}
	sourceTable := &SimpleTableSource{
		Token: p.curToken,
		Name: &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		},
	}
	stmt.SourceTable = sourceTable

	// Check for AS keyword and source alias
	if p.peekToken.Type != TokenEOF && p.peekTokenIsKeyword("AS") {
		p.nextToken() // consume AS
		if !p.expectPeek(TokenIdentifier) {
			p.addError(fmt.Sprintf("expected source table alias after AS, got %s at %s",
				p.peekToken.Literal, p.peekToken.Position))
			return nil
		}
		stmt.SourceAlias = &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		}
		sourceTable.Alias = stmt.SourceAlias
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file, expected ON at %s", p.curToken.Position))
		return nil
	}

	// Expect ON
	if !p.expectKeyword("ON") {
		return nil
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file after ON at %s", p.curToken.Position))
		return nil
	}

	// Parse join condition
	p.nextToken() // move to the first token of the condition
	stmt.OnCondition = p.parseExpression(LOWEST)
	if stmt.OnCondition == nil {
		p.addError(fmt.Sprintf("expected merge condition after ON at %s", p.curToken.Position))
		return nil
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file, expected WHEN at %s", p.curToken.Position))
		return nil
	}

	// Parse WHEN clauses
	for p.peekToken.Type != TokenEOF && p.peekTokenIsKeyword("WHEN") {
		clause := p.parseMergeClause()
		if clause == nil {
			return nil
		}
		stmt.Clauses = append(stmt.Clauses, clause)
	}

	// Ensure we have at least one clause
	if len(stmt.Clauses) == 0 {
		p.addError(fmt.Sprintf("expected at least one WHEN clause at %s", p.curToken.Position))
		return nil
	}

	return stmt
}

// parseMergeClause parses a WHEN [NOT] MATCHED THEN ... clause in a MERGE statement
func (p *Parser) parseMergeClause() *MergeClause {
	p.nextToken() // consume WHEN

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file after WHEN at %s", p.curToken.Position))
		return nil
	}

	// Create a new merge clause
	clause := &MergeClause{
		IsMatched: true, // Default to MATCHED (will be set to false if NOT is found)
	}

	// Check for NOT
	if p.peekToken.Type != TokenEOF && p.peekTokenIsKeyword("NOT") {
		p.nextToken() // consume NOT
		clause.IsMatched = false
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file, expected MATCHED at %s", p.curToken.Position))
		return nil
	}

	// Expect MATCHED
	if !p.expectKeyword("MATCHED") {
		return nil
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file after MATCHED at %s", p.curToken.Position))
		return nil
	}

	// Check for optional AND condition
	if p.peekToken.Type != TokenEOF && p.peekTokenIsKeyword("AND") {
		p.nextToken() // consume AND

		// Make sure we have tokens to process
		if p.peekToken.Type == TokenEOF {
			p.addError(fmt.Sprintf("unexpected end of file after AND at %s", p.curToken.Position))
			return nil
		}

		p.nextToken() // move to the first token of the condition

		clause.Condition = p.parseExpression(LOWEST)
		if clause.Condition == nil {
			p.addError(fmt.Sprintf("expected condition after AND at %s", p.curToken.Position))
			return nil
		}
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file, expected THEN at %s", p.curToken.Position))
		return nil
	}

	// Expect THEN
	if !p.expectKeyword("THEN") {
		return nil
	}

	// Make sure we have tokens to process
	if p.peekToken.Type == TokenEOF {
		p.addError(fmt.Sprintf("unexpected end of file after THEN at %s", p.curToken.Position))
		return nil
	}

	// Parse the action (UPDATE, INSERT, or DELETE)
	if !p.expectPeek(TokenKeyword) {
		p.addError(fmt.Sprintf("expected UPDATE, INSERT, or DELETE after THEN, got %s at %s",
			p.peekToken.Literal, p.peekToken.Position))
		return nil
	}

	action := strings.ToUpper(p.curToken.Literal)
	switch action {
	case "UPDATE":
		// Process UPDATE action
		clause.Action = ActionUpdate
		clause.Assignments = make(map[string]Expression)

		// Make sure we have tokens to process
		if p.peekToken.Type == TokenEOF {
			p.addError(fmt.Sprintf("unexpected end of file after UPDATE, expected SET at %s", p.curToken.Position))
			return nil
		}

		// Expect SET
		if !p.expectKeyword("SET") {
			return nil
		}

		// Parse column-value pairs
		for {
			// Make sure we have tokens to process
			if p.peekToken.Type == TokenEOF {
				p.addError(fmt.Sprintf("unexpected end of file, expected column name at %s", p.curToken.Position))
				return nil
			}

			// Parse column name
			if !p.expectPeek(TokenIdentifier) {
				p.addError(fmt.Sprintf("expected column name, got %s at %s",
					p.peekToken.Literal, p.peekToken.Position))
				return nil
			}
			columnName := p.curToken.Literal

			// Make sure we have tokens to process
			if p.peekToken.Type == TokenEOF {
				p.addError(fmt.Sprintf("unexpected end of file, expected = at %s", p.curToken.Position))
				return nil
			}

			// Expect equals sign
			if !p.expectPeek(TokenOperator) || p.curToken.Literal != "=" {
				p.addError(fmt.Sprintf("expected '=', got %s at %s", p.curToken.Literal, p.curToken.Position))
				return nil
			}

			// Make sure we have tokens to process
			if p.peekToken.Type == TokenEOF {
				p.addError(fmt.Sprintf("unexpected end of file after =, expected value expression at %s", p.curToken.Position))
				return nil
			}

			// Parse value expression
			p.nextToken() // move to the value expression
			valueExpr := p.parseExpression(LOWEST)
			if valueExpr == nil {
				p.addError(fmt.Sprintf("expected expression after '=' at %s", p.curToken.Position))
				return nil
			}

			// Add to assignments map
			clause.Assignments[columnName] = valueExpr

			// Check for comma (for multiple assignments)
			if p.peekToken.Type == TokenEOF || !p.peekTokenIsPunctuator(",") {
				break
			}
			p.nextToken() // consume comma
		}

	case "INSERT":
		// Process INSERT action
		clause.Action = ActionInsert

		// Check for optional column list
		if p.peekTokenIsPunctuator("(") {
			p.nextToken() // consume (
			clause.Columns = p.parseIdentifierList()

			// Expect closing parenthesis
			if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
				p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
				return nil
			}
		}

		// Expect VALUES
		if !p.expectKeyword("VALUES") {
			return nil
		}

		// Expect opening parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
			p.addError(fmt.Sprintf("expected '(', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		// Parse value list
		p.nextToken() // move to first expression
		for {
			expr := p.parseExpression(LOWEST)
			if expr == nil {
				p.addError(fmt.Sprintf("expected expression in values list at %s", p.curToken.Position))
				return nil
			}

			clause.Values = append(clause.Values, expr)

			// Check for comma
			if !p.peekTokenIsPunctuator(",") {
				break
			}
			p.nextToken() // consume comma
			p.nextToken() // move to next expression
		}

		// Expect closing parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

	case "DELETE":
		// Process DELETE action
		clause.Action = ActionDelete
		// No further parsing needed for DELETE

	default:
		p.addError(fmt.Sprintf("expected UPDATE, INSERT or DELETE, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	return clause
}

// parseInsertStatement parses an INSERT statement
func (p *Parser) parseInsertStatement() Statement {
	// Create an INSERT statement with the current token
	stmt := &InsertStatement{
		Token: p.curToken,
	}

	// Expect INTO
	if !p.expectKeyword("INTO") {
		return nil
	}

	// Parse table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Check for column list
	if p.peekTokenIsPunctuator("(") {
		p.nextToken() // consume (
		stmt.Columns = p.parseIdentifierList()

		// Expect closing parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}
	}

	// Expect VALUES
	if !p.expectKeyword("VALUES") {
		return nil
	}

	// Parse value lists
	stmt.Values = p.parseValueLists()

	// Check for ON DUPLICATE KEY UPDATE clause
	if p.peekTokenIsKeyword("ON") {
		p.nextToken() // consume ON

		if !p.expectKeyword("DUPLICATE") {
			return nil
		}

		if !p.expectKeyword("KEY") {
			return nil
		}

		if !p.expectKeyword("UPDATE") {
			return nil
		}

		// Parse the update assignments
		stmt.OnDuplicate = true
		stmt.UpdateColumns = []*Identifier{}
		stmt.UpdateExpressions = []Expression{}

		for {
			// Parse column to update
			if !p.expectPeek(TokenIdentifier) {
				return nil
			}

			column := &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			}
			stmt.UpdateColumns = append(stmt.UpdateColumns, column)

			// Expect =
			if !p.expectPeek(TokenOperator) || p.curToken.Literal != "=" {
				p.addError(fmt.Sprintf("expected '=', got %s at %s", p.curToken.Literal, p.curToken.Position))
				return nil
			}

			// Parse expression
			p.nextToken() // consume =
			expr := p.parseExpression(LOWEST)
			if expr == nil {
				return nil
			}
			stmt.UpdateExpressions = append(stmt.UpdateExpressions, expr)

			// Check for comma (more assignments) or end
			if !p.peekTokenIsPunctuator(",") {
				break
			}
			p.nextToken() // consume ,
		}
	}

	return stmt
}

// parseUpdateStatement parses an UPDATE statement
func (p *Parser) parseUpdateStatement() *UpdateStatement {
	// Create an UPDATE statement with the current token
	stmt := &UpdateStatement{
		Token:   p.curToken,
		Updates: make(map[string]Expression),
	}

	// Parse table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Expect SET keyword
	if !p.expectKeyword("SET") {
		return nil
	}

	// Parse column-value pairs
	for {
		// Parse column name
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}
		columnName := p.curToken.Literal

		// Expect equal sign
		if !p.expectPeek(TokenOperator) || p.curToken.Literal != "=" {
			p.addError(fmt.Sprintf("expected '=', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		// Parse value expression
		p.nextToken() // move to the value expression

		// If we're dealing with "salary * 1.1" type of expression, manually parse it
		if p.curTokenIs(TokenIdentifier) && p.peekTokenIsOperator("*") {
			// Save the identifier
			left := &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			}

			// Consume the * operator
			p.nextToken() // move to *

			// Create an infix expression node
			expr := &InfixExpression{
				Token:    p.curToken,
				Operator: p.curToken.Literal,
				Left:     left,
			}

			// Parse the right side (should be a numeric literal)
			p.nextToken() // move to the number

			if p.curTokenIs(TokenInteger) {
				value, _ := strconv.ParseInt(p.curToken.Literal, 0, 64)
				expr.Right = &IntegerLiteral{
					Token: p.curToken,
					Value: value,
				}
			} else if p.curTokenIs(TokenFloat) {
				value, _ := strconv.ParseFloat(p.curToken.Literal, 64)
				expr.Right = &FloatLiteral{
					Token: p.curToken,
					Value: value,
				}
			} else {
				p.addError(fmt.Sprintf("expected number after '*', got %s", p.tokenDebug(p.curToken)))
				return nil
			}

			// Add the completed expression
			stmt.Updates[columnName] = expr
		} else {
			// Standard expression parsing
			valueExpr := p.parseExpression(LOWEST)
			if valueExpr == nil {
				p.addError(fmt.Sprintf("expected expression after '=' at %s", p.curToken.Position))
				return nil
			}

			// Add to updates map
			stmt.Updates[columnName] = valueExpr
		}

		// Check for comma (for multiple updates)
		if !p.peekTokenIsPunctuator(",") {
			break
		}
		p.nextToken() // consume comma
	}

	// Parse WHERE clause if present
	if p.peekTokenIsKeyword("WHERE") {
		p.nextToken() // consume WHERE
		p.nextToken() // move to the expression

		// Save current clause and set to WHERE
		prevClause := p.currentClause
		p.currentClause = "WHERE"

		stmt.Where = p.parseExpression(LOWEST)
		if stmt.Where == nil {
			p.addError(fmt.Sprintf("expected expression after WHERE at %s", p.curToken.Position))
			return nil
		}

		// Restore previous clause
		p.currentClause = prevClause
	}

	return stmt
}

// parseDeleteStatement parses a DELETE statement
func (p *Parser) parseDeleteStatement() *DeleteStatement {
	// Create a DELETE statement with the current token
	stmt := &DeleteStatement{
		Token: p.curToken,
	}

	// Expect FROM keyword
	if !p.expectKeyword("FROM") {
		return nil
	}

	// Parse table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Parse WHERE clause if present
	if p.peekTokenIsKeyword("WHERE") {
		p.nextToken() // consume WHERE
		p.nextToken() // move to the expression

		// Save current clause and set to WHERE
		prevClause := p.currentClause
		p.currentClause = "WHERE"

		stmt.Where = p.parseExpression(LOWEST)

		// Restore previous clause
		p.currentClause = prevClause
	}

	return stmt
}

// parseAlterStatement parses an ALTER statement
func (p *Parser) parseAlterStatement() Statement {
	// Create an ALTER statement with the current token
	token := p.curToken

	// Check what we're altering
	if !p.expectKeyword("TABLE") {
		return nil
	}

	// Initialize ALTER TABLE statement
	stmt := &AlterTableStatement{
		Token: token,
	}

	// Parse table name
	if !p.expectPeek(TokenIdentifier) {
		return nil
	}
	stmt.TableName = &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}

	// Ensure we have another token
	if !p.expectPeek(TokenKeyword) {
		p.addError(fmt.Sprintf("expected ALTER TABLE operation, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse the alter operation based on the current keyword
	switch strings.ToUpper(p.curToken.Literal) {
	case "ADD":
		stmt.Operation = AddColumn

		// Check for optional COLUMN keyword
		if p.peekTokenIsKeyword("COLUMN") {
			p.nextToken() // consume COLUMN
		}

		// Parse column definition
		p.nextToken() // move to column name
		if !p.curTokenIs(TokenIdentifier) {
			p.addError(fmt.Sprintf("expected column name, got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		columnDef := p.parseColumnDefinition()
		stmt.ColumnDef = &columnDef

	case "DROP":
		stmt.Operation = DropColumn

		// Check for optional COLUMN keyword
		if p.peekTokenIsKeyword("COLUMN") {
			p.nextToken() // consume COLUMN
		}

		// Parse column name
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}
		stmt.ColumnName = &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		}

	case "RENAME":
		// Check next keyword to determine whether we're renaming a column or the table
		if !p.expectPeek(TokenKeyword) {
			return nil
		}

		if strings.ToUpper(p.curToken.Literal) == "COLUMN" {
			stmt.Operation = RenameColumn

			// Parse old column name
			if !p.expectPeek(TokenIdentifier) {
				return nil
			}
			stmt.ColumnName = &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			}

			// Expect TO keyword
			if !p.expectKeyword("TO") {
				return nil
			}

			// Parse new column name
			if !p.expectPeek(TokenIdentifier) {
				return nil
			}
			stmt.NewColumnName = &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			}
		} else if strings.ToUpper(p.curToken.Literal) == "TO" {
			stmt.Operation = RenameTable

			// Parse new table name
			if !p.expectPeek(TokenIdentifier) {
				return nil
			}
			stmt.NewTableName = &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			}
		} else {
			p.addError(fmt.Sprintf("expected COLUMN or TO after RENAME, got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

	case "MODIFY":
		stmt.Operation = ModifyColumn

		// Check for optional COLUMN keyword
		if p.peekTokenIsKeyword("COLUMN") {
			p.nextToken() // consume COLUMN
		}

		// Parse column definition
		p.nextToken() // move to column name
		if !p.curTokenIs(TokenIdentifier) {
			p.addError(fmt.Sprintf("expected column name, got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		columnDef := p.parseColumnDefinition()
		stmt.ColumnDef = &columnDef

	default:
		p.addError(fmt.Sprintf("expected ALTER TABLE operation (ADD, DROP, RENAME, MODIFY), got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	return stmt
}

// parseIdentifierList parses a comma-separated list of identifiers
func (p *Parser) parseIdentifierList() []*Identifier {
	list := []*Identifier{}

	p.nextToken() // Move to the first identifier
	list = append(list, &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	})

	for p.peekTokenIsPunctuator(",") {
		p.nextToken() // consume comma
		if !p.expectPeek(TokenIdentifier) {
			return list
		}
		list = append(list, &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		})
	}

	return list
}

// parseValueLists parses the VALUES part of an INSERT statement
func (p *Parser) parseValueLists() [][]Expression {
	valueLists := [][]Expression{}

	// Expect opening parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
		p.addError(fmt.Sprintf("expected '(', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse first value list
	valueList := p.parseExpressionList()
	valueLists = append(valueLists, valueList)

	// Expect closing parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
		p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse additional value lists
	for p.peekTokenIsPunctuator(",") {
		p.nextToken() // consume comma

		// Expect opening parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
			p.addError(fmt.Sprintf("expected '(', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		valueList = p.parseExpressionList()
		valueLists = append(valueLists, valueList)

		// Expect closing parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}
	}

	return valueLists
}

// parseExpression parses an expression with the given precedence
func (p *Parser) parseExpression(precedence int) Expression {
	// Get the prefix parse function for the current token
	prefix := p.prefixParseFns[p.curToken.Type]
	if prefix == nil {
		p.addError(fmt.Sprintf("no prefix parse function for %s at %s", p.curToken.Type, p.curToken.Position))
		return nil
	}

	// Parse the prefix expression
	leftExp := prefix()
	if leftExp == nil {
		return nil
	}

	// Keep parsing infix expressions as long as the next token has a higher precedence
	for !p.peekTokenIs(TokenEOF) && precedence < p.peekPrecedence() {
		infix := p.infixParseFns[p.peekToken.Type]
		if infix == nil {
			return leftExp
		}

		p.nextToken()
		nextExp := infix(leftExp)
		if nextExp == nil {
			return nil
		}
		leftExp = nextExp
	}

	return leftExp
}

// parseIdentifier parses an identifier
func (p *Parser) parseIdentifier() Expression {
	return &Identifier{
		Token: p.curToken,
		Value: p.curToken.Literal,
	}
}

// parseIntegerLiteral parses an integer literal
func (p *Parser) parseIntegerLiteral() Expression {
	lit := &IntegerLiteral{Token: p.curToken}

	// Convert the token literal to an integer
	value, err := strconv.ParseInt(p.curToken.Literal, 0, 64)
	if err != nil {
		p.addError(fmt.Sprintf("could not parse %q as integer: %s", p.curToken.Literal, err))
		return nil
	}

	lit.Value = value
	return lit
}

// parseFloatLiteral parses a float literal
func (p *Parser) parseFloatLiteral() Expression {
	lit := &FloatLiteral{Token: p.curToken}

	// Convert the token literal to a float
	value, err := strconv.ParseFloat(p.curToken.Literal, 64)
	if err != nil {
		p.addError(fmt.Sprintf("could not parse %q as float: %s", p.curToken.Literal, err))
		return nil
	}

	lit.Value = value
	return lit
}

// parseParameter parses a parameter token
func (p *Parser) parseParameter() Expression {
	param := &Parameter{
		Token: p.curToken,
		Name:  p.curToken.Literal,
	}

	// Set the index based on the parameter style
	if param.Name == "?" {
		// For ? style parameters, we'll assign the index later when we build the prepared statement
		param.Index = 0
	} else if len(param.Name) > 1 && param.Name[0] == '$' {
		// For $N style parameters, extract N
		indexStr := param.Name[1:]
		index, err := strconv.Atoi(indexStr)
		if err != nil {
			p.addError(fmt.Sprintf("invalid parameter index %q: %s", indexStr, err))
			return nil
		}
		param.Index = index
	} else {
		// This should never happen as the lexer should catch it
		p.addError(fmt.Sprintf("invalid parameter format: %s", param.Name))
		return nil
	}

	// Add contextual information about the parameter's location
	param.Location = p.currentClause

	// Assign a unique ID for the statement
	// Each statement in a multi-statement query should have a different ID
	param.StatementID = p.currentStatementID

	// Assign a position within the current statement
	// This helps maintain the order of parameters within a statement
	param.OrderInStatement = p.parameterCounter
	p.parameterCounter++ // Increment for the next parameter
	p.parameterStatementCounter++

	return param
}

// parseStringLiteral parses a string literal
func (p *Parser) parseStringLiteral() Expression {
	// Extract the content of the string without quotes
	literal := p.curToken.Literal

	// Check if the string has opening and closing quotes
	if len(literal) >= 2 {
		value := literal[1 : len(literal)-1] // Remove quotes

		// Handle escape sequences
		value = strings.Replace(value, "\\n", "\n", -1)
		value = strings.Replace(value, "\\t", "\t", -1)
		value = strings.Replace(value, "\\r", "\r", -1)
		value = strings.Replace(value, "\\\"", "\"", -1)
		value = strings.Replace(value, "\\'", "'", -1)
		value = strings.Replace(value, "\\\\", "\\", -1)

		// Create the string literal
		strLiteral := &StringLiteral{
			Token: p.curToken,
			Value: value,
		}

		// Add type hints based on content for special types
		if len(value) > 0 && value[0] == '{' && value[len(value)-1] == '}' {
			// Check for JSON format (starts with { and ends with })
			strLiteral.TypeHint = "JSON"
		}

		return strLiteral
	}

	// No quotes or malformed string
	p.addError(fmt.Sprintf("malformed string literal: %s at %s", literal, p.curToken.Position))
	return &StringLiteral{
		Token: p.curToken,
		Value: literal,
	}
}

// parseKeyword parses a keyword
func (p *Parser) parseKeyword() Expression {
	switch strings.ToUpper(p.curToken.Literal) {
	case "TRUE", "FALSE":
		return &BooleanLiteral{
			Token: p.curToken,
			Value: strings.ToUpper(p.curToken.Literal) == "TRUE",
		}
	case "NULL":
		return &NullLiteral{
			Token: p.curToken,
		}
	case "CASE":
		return p.parseCaseExpression()
	case "CAST":
		// CAST is treated as a special expression with its own syntax
		if p.peekTokenIsPunctuator("(") {
			return p.parseCastExpression()
		}
		// If not followed by a parenthesis, handle as an error
		p.addError(fmt.Sprintf("expected '(' after CAST keyword at %s", p.curToken.Position))
		return nil
	case "INTERVAL":
		return p.parseIntervalLiteral()
	case "TIMESTAMP", "DATE", "TIME":
		// Handle type literals like TIMESTAMP '2025-01-01'
		if p.peekTokenIs(TokenString) {
			typeHint := p.curToken.Literal
			p.nextToken() // consume the string

			// Extract the string value without quotes
			literal := p.curToken.Literal
			value := literal
			if len(literal) >= 2 && literal[0] == '\'' && literal[len(literal)-1] == '\'' {
				value = literal[1 : len(literal)-1] // Remove quotes
			}

			return &StringLiteral{
				Token:    p.curToken,
				Value:    value,
				TypeHint: typeHint,
			}
		}
		// If not followed by a string, handle as an error
		p.addError(fmt.Sprintf("expected string literal after %s keyword at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	case "EXISTS":
		// EXISTS is handled as a special expression
		return p.parseExistsExpression()
	default:
		p.addError(fmt.Sprintf("unexpected keyword: %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}
}

// parseIntervalLiteral parses an INTERVAL literal (e.g. INTERVAL '1 day')
func (p *Parser) parseIntervalLiteral() Expression {
	intervalToken := p.curToken

	// Expect a string literal after INTERVAL
	if !p.expectPeek(TokenString) {
		return nil
	}

	// Extract the string value without quotes
	literal := p.curToken.Literal
	value := literal
	if len(literal) >= 2 && literal[0] == '\'' && literal[len(literal)-1] == '\'' {
		value = literal[1 : len(literal)-1] // Remove quotes
	}

	// Parse the interval value to extract quantity and unit
	parts := strings.Fields(value)
	if len(parts) < 2 {
		p.addError(fmt.Sprintf("invalid interval format: %s at %s", value, p.curToken.Position))
		return nil
	}

	// Parse the numeric quantity
	quantity := int64(0)
	if num, err := strconv.ParseInt(parts[0], 10, 64); err == nil {
		quantity = num
	} else {
		p.addError(fmt.Sprintf("invalid interval quantity: %s at %s", parts[0], p.curToken.Position))
		return nil
	}

	// Get the unit (normalize to lowercase)
	unit := strings.ToLower(parts[1])

	// Normalize plural units to singular
	unit = strings.TrimSuffix(unit, "s")

	return &IntervalLiteral{
		Token:    intervalToken,
		Value:    value,
		Quantity: quantity,
		Unit:     unit,
	}
}

// parseCaseExpression parses a CASE expression
func (p *Parser) parseCaseExpression() Expression {
	// Create a CASE expression with the current token
	expression := &CaseExpression{
		Token:       p.curToken,
		WhenClauses: []*WhenClause{},
	}

	// Check if this is a simple CASE expression (with a value to match)
	// or a searched CASE expression (with conditions)
	if !p.peekTokenIsKeyword("WHEN") {
		// This is a simple CASE expression, so parse the value
		p.nextToken() // Move to the value expression
		expression.Value = p.parseExpression(LOWEST)
		if expression.Value == nil {
			p.addError(fmt.Sprintf("expected expression after CASE at %s", p.curToken.Position))
			return nil
		}
	}

	// Parse WHEN clauses
	for p.peekTokenIsKeyword("WHEN") {
		p.nextToken() // Consume WHEN
		whenToken := p.curToken

		// Parse the condition
		p.nextToken() // Move to the condition expression
		condition := p.parseExpression(LOWEST)
		if condition == nil {
			p.addError(fmt.Sprintf("expected condition after WHEN at %s", p.curToken.Position))
			return nil
		}

		// Expect THEN
		if !p.expectKeyword("THEN") {
			return nil
		}

		// Parse the result expression
		p.nextToken() // Move to the result expression
		result := p.parseExpression(LOWEST)
		if result == nil {
			p.addError(fmt.Sprintf("expected result expression after THEN at %s", p.curToken.Position))
			return nil
		}

		// Create a WHEN clause and add it to the CASE expression
		whenClause := &WhenClause{
			Token:      whenToken,
			Condition:  condition,
			ThenResult: result,
		}
		expression.WhenClauses = append(expression.WhenClauses, whenClause)
	}

	// There must be at least one WHEN clause
	if len(expression.WhenClauses) == 0 {
		p.addError(fmt.Sprintf("expected at least one WHEN clause in CASE expression at %s", p.curToken.Position))
		return nil
	}

	// Check for ELSE clause
	if p.peekTokenIsKeyword("ELSE") {
		p.nextToken() // Consume ELSE

		// Parse the ELSE expression
		p.nextToken() // Move to the ELSE expression
		expression.ElseValue = p.parseExpression(LOWEST)
		if expression.ElseValue == nil {
			p.addError(fmt.Sprintf("expected expression after ELSE at %s", p.curToken.Position))
			return nil
		}
	}

	// Expect END
	if !p.expectKeyword("END") {
		return nil
	}

	return expression
}

// parsePrefixExpression parses a prefix expression
func (p *Parser) parsePrefixExpression() Expression {
	// Handle EXISTS keyword
	if p.curTokenIsKeyword("EXISTS") {
		return p.parseExistsExpression()
	}

	// Handle NOT keyword for NOT EXISTS and NOT IN
	if p.curTokenIsKeyword("NOT") {
		if p.peekTokenIsKeyword("EXISTS") {
			// NOT EXISTS
			p.nextToken() // consume EXISTS
			existsExpr := p.parseExistsExpression()
			if existsExpr == nil {
				return nil
			}

			// Wrap the EXISTS expression in a NOT prefix expression
			return &PrefixExpression{
				Token:    p.curToken,
				Operator: "NOT",
				Right:    existsExpr,
			}
		} else if p.peekTokenIsKeyword("IN") {
			// NOT IN is handled in parseInExpression directly
			p.addError(fmt.Sprintf("NOT IN should be handled by the IN keyword parsing at %s", p.curToken.Position))
			return nil
		}
	}

	// Create a prefix expression with the current token
	expression := &PrefixExpression{
		Token:    p.curToken,
		Operator: p.curToken.Literal,
	}

	// Parse the right expression
	p.nextToken()

	// Special handling for negative numbers
	if p.curTokenIs(TokenInteger) || p.curTokenIs(TokenFloat) {
		if expression.Operator == "-" {
			if p.curTokenIs(TokenInteger) {
				value, _ := strconv.ParseInt(p.curToken.Literal, 0, 64)
				return &IntegerLiteral{
					Token: p.curToken,
					Value: -value,
				}
			} else {
				value, _ := strconv.ParseFloat(p.curToken.Literal, 64)
				return &FloatLiteral{
					Token: p.curToken,
					Value: -value,
				}
			}
		}
	}

	expression.Right = p.parseExpression(PREFIX)
	return expression
}

// parseExistsExpression parses an EXISTS expression
func (p *Parser) parseExistsExpression() Expression {
	// Create an EXISTS expression
	expression := &ExistsExpression{
		Token: p.curToken,
	}

	// Expect opening parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
		p.addError(fmt.Sprintf("expected '(' after EXISTS, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Parse the SELECT statement
	p.nextToken() // Move to SELECT
	if !p.curTokenIsKeyword("SELECT") {
		p.addError(fmt.Sprintf("expected SELECT in EXISTS subquery, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	expression.Subquery = p.parseSelectStatement()
	if expression.Subquery == nil {
		return nil
	}

	// Expect closing parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
		p.addError(fmt.Sprintf("expected ')' after EXISTS subquery, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	return expression
}

// parsePunctuator parses a punctuator
func (p *Parser) parsePunctuator() Expression {
	switch p.curToken.Literal {
	case "(":
		// Save the token for possible subquery
		token := p.curToken

		// Check if this might be a subquery (scalar subquery in an expression)
		if p.peekTokenIsKeyword("SELECT") {
			p.nextToken() // Move to SELECT
			selectStmt := p.parseSelectStatement()
			if selectStmt == nil {
				return nil
			}

			// Expect closing parenthesis
			if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
				p.addError(fmt.Sprintf("expected ')' after scalar subquery, got %s at %s", p.curToken.Literal, p.curToken.Position))
				return nil
			}

			// Create and return a scalar subquery
			return &ScalarSubquery{
				Token:    token,
				Subquery: selectStmt,
			}
		}

		// Regular group expression
		p.nextToken() // consume (

		// Empty parentheses
		if p.curTokenIsPunctuator(")") {
			p.nextToken() // consume )
			// For now, just return an identifier with empty value
			return &Identifier{
				Token: p.curToken,
				Value: "()",
			}
		}

		exp := p.parseExpression(LOWEST)
		if exp == nil {
			p.addError(fmt.Sprintf("expected expression after '(' at %s", p.curToken.Position))
			return nil
		}

		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		return exp
	case "*":
		// * can be a wildcard in SELECT statements
		return &Identifier{
			Token: p.curToken,
			Value: "*",
		}
	default:
		p.addError(fmt.Sprintf("unexpected punctuator: %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}
}

// parseInfixExpression parses an infix expression
func (p *Parser) parseInfixExpression(left Expression) Expression {
	// Create an infix expression with the current token
	expression := &InfixExpression{
		Token:    p.curToken,
		Operator: p.curToken.Literal,
		Left:     left,
	}

	// Get the precedence of the current operator
	precedence := p.curPrecedence()

	// Parse the right expression with the correct precedence
	p.nextToken()

	// Check for nil right expression
	right := p.parseExpression(precedence)
	if right == nil {
		p.addError(fmt.Sprintf("missing right-hand side of operator %s at %s", expression.Operator, expression.Token.Position))
		return nil
	}

	expression.Right = right
	return expression
}

// parseInfixKeyword parses an infix expression with a keyword operator
func (p *Parser) parseInfixKeyword(left Expression) Expression {
	switch strings.ToUpper(p.curToken.Literal) {
	case "AND", "OR", "LIKE":
		// Create an infix expression with the current token
		expression := &InfixExpression{
			Token:    p.curToken,
			Operator: strings.ToUpper(p.curToken.Literal),
			Left:     left,
		}

		// Get the precedence of the current operator
		precedence, ok := precedences[strings.ToUpper(p.curToken.Literal)]
		if !ok {
			precedence = EQUALS // Default to EQUALS precedence for unknown operators
		}

		// Parse the right expression with the correct precedence
		p.nextToken()
		expression.Right = p.parseExpression(precedence)

		return expression

	case "IS":
		// Create an IS expression token
		token := p.curToken
		operator := "IS" // Default operator is "IS"

		// Check for "IS NOT" (peek ahead for NOT)
		if p.peekTokenIsKeyword("NOT") {
			p.nextToken() // consume NOT
			operator = "IS NOT"
		}

		// Next token should be NULL
		p.nextToken()
		if !p.curTokenIsKeyword("NULL") {
			p.addError(fmt.Sprintf("expected NULL after IS operator, got %s at %s",
				p.curToken.Literal, p.curToken.Position))
			return nil
		}

		// Create an IS expression (InfixExpression with IS operator)
		return &InfixExpression{
			Token:    token,
			Operator: operator,
			Left:     left,
			Right:    &NullLiteral{Token: p.curToken},
		}

	case "BETWEEN":
		// Create a BETWEEN expression
		token := p.curToken

		// Parse the lower bound
		p.nextToken()
		lower := p.parseExpression(EQUALS)

		// Expect AND
		if !p.expectPeek(TokenKeyword) || !strings.EqualFold(p.curToken.Literal, "AND") {
			p.addError(fmt.Sprintf("expected 'AND' after BETWEEN expression, got %s at %s",
				p.curToken.Literal, p.curToken.Position))
			return nil
		}

		// Parse the upper bound
		p.nextToken()
		upper := p.parseExpression(EQUALS)

		// Return a new BetweenExpression
		return &BetweenExpression{
			Token: token,
			Expr:  left,
			Lower: lower,
			Upper: upper,
			Not:   false,
		}
	case "AS":
		// Create an aliased expression
		expression := &AliasedExpression{
			Token:      p.curToken,
			Expression: left,
		}

		// Parse the alias
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}

		expression.Alias = &Identifier{
			Token: p.curToken,
			Value: p.curToken.Literal,
		}

		return expression
	case "IN":
		return p.parseInExpression(left, false)
	case "NOT":
		// Handle NOT IN
		if p.peekTokenIsKeyword("IN") {
			notToken := p.curToken // Save the NOT token
			p.nextToken()          // Consume the IN token

			// Create an IN expression but make sure to use the NOT token for accurate position
			inExpr := p.parseInExpression(left, true)

			// Override the token in the returned expression to point to NOT instead of IN
			if inExpr, ok := inExpr.(*InExpression); ok {
				inExpr.Token = notToken
			}

			return inExpr
		}

		// For other NOT expressions, handle as a prefix for the right-side expression
		// Create a prefix expression with the current token
		token := p.curToken
		p.nextToken() // consume NOT

		// Parse the expression that follows NOT
		expr := p.parseExpression(PREFIX)

		// Create a NOT infix expression (NOT is an infix operator here)
		return &InfixExpression{
			Token:    token,
			Operator: "NOT",
			Left:     left,
			Right:    expr,
		}
	default:
		p.addError(fmt.Sprintf("unexpected infix keyword: %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}
}

// parseInExpression parses an IN expression
func (p *Parser) parseInExpression(left Expression, not bool) Expression {
	// Create an IN expression
	expression := &InExpression{
		Token: p.curToken,
		Left:  left,
		Not:   not,
	}

	// Expect opening parenthesis
	if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != "(" {
		p.addError(fmt.Sprintf("expected '(' after IN, got %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}

	// Check if this is a subquery or a list of values
	if p.peekTokenIsKeyword("SELECT") {
		// This is a subquery
		p.nextToken() // Move to SELECT

		selectStmt := p.parseSelectStatement()
		if selectStmt == nil {
			return nil
		}

		// Create a scalar subquery
		expression.Right = &ScalarSubquery{
			Token:    p.curToken,
			Subquery: selectStmt,
		}

		// Expect closing parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected ')' after IN subquery, got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}
	} else {
		// This is a list of values
		token := p.curToken // Save the opening parenthesis token

		// Parse the first expression
		p.nextToken() // Move past the opening parenthesis

		exprList := &ExpressionList{
			Token:       token,
			Expressions: []Expression{},
		}

		// Handle empty list
		if p.curTokenIsPunctuator(")") {
			// Just return the empty list
			expression.Right = exprList
			return expression
		}

		// Parse first expression
		expr := p.parseExpression(LOWEST)
		if expr == nil {
			return nil
		}
		exprList.Expressions = append(exprList.Expressions, expr)

		// Parse additional expressions
		for p.peekTokenIsPunctuator(",") {
			p.nextToken() // Consume the comma
			p.nextToken() // Move to the next expression

			expr = p.parseExpression(LOWEST)
			if expr == nil {
				return nil
			}
			exprList.Expressions = append(exprList.Expressions, expr)
		}

		// Expect closing parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected ')' or ',' in IN expression, got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		expression.Right = exprList
	}

	return expression
}

// parseInfixPunctuator parses an infix expression with a punctuator operator
func (p *Parser) parseInfixPunctuator(left Expression) Expression {
	switch p.curToken.Literal {
	case ".":
		// Table.Column reference
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}

		// Make sure the left side is an identifier
		leftIdent, ok := left.(*Identifier)
		if !ok {
			p.addError(fmt.Sprintf("left side of '.' must be an identifier, got %T at %s", left, p.curToken.Position))
			return nil
		}

		return &QualifiedIdentifier{
			Token:     leftIdent.Token,
			Qualifier: leftIdent,
			Name: &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			},
		}
	case "(":
		// Function call
		leftIdent, ok := left.(*Identifier)
		if !ok {
			p.addError(fmt.Sprintf("left side of '(' must be an identifier, got %T at %s", left, p.curToken.Position))
			return nil
		}

		// Create function call node
		call := &FunctionCall{
			Token:     leftIdent.Token,
			Function:  leftIdent.Value,
			Arguments: []Expression{},
		}

		// Check for empty parameters - e.g., COUNT()
		if p.peekTokenIsPunctuator(")") {
			p.nextToken() // consume )
			return call
		}

		// Special handling for * argument (e.g., COUNT(*))
		if p.peekTokenIsOperator("*") {
			p.nextToken() // consume *

			// Add star as argument
			call.Arguments = append(call.Arguments, &Identifier{
				Token: p.curToken,
				Value: "*",
			})

			// Check for ORDER BY clause inside the function
			if p.peekTokenIsKeyword("ORDER") {
				p.nextToken() // consume ORDER

				// Expect BY keyword
				if !p.expectKeyword("BY") {
					return nil
				}

				p.nextToken() // move to first expression

				// Parse first ORDER BY expression
				orderByExpr := p.parseOrderByExpression()
				if orderByExpr.Expression != nil {
					call.OrderBy = append(call.OrderBy, orderByExpr)
				}

				// Parse additional ORDER BY expressions separated by commas
				for p.peekTokenIsPunctuator(",") {
					p.nextToken() // consume comma
					p.nextToken() // move to next ORDER BY expression

					orderByExpr := p.parseOrderByExpression()
					call.OrderBy = append(call.OrderBy, orderByExpr)
				}
			}

			// Expect closing parenthesis
			if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
				p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
				return nil
			}

			return call
		}

		// Special handling for DISTINCT keyword
		if p.peekTokenIsKeyword("DISTINCT") {
			distToken := p.peekToken
			p.nextToken() // consume DISTINCT

			// Mark the function call as using DISTINCT
			call.IsDistinct = true

			// Next token should be column name or expression
			p.nextToken()

			// Parse the expression that follows DISTINCT
			expr := p.parseExpression(LOWEST)
			if expr == nil {
				p.addError(fmt.Sprintf("expected expression after DISTINCT at %s", distToken.Position))
				return nil
			}

			// Add the expression as the first argument
			call.Arguments = append(call.Arguments, expr)

			// Check for additional arguments after comma
			for p.peekTokenIsPunctuator(",") {
				p.nextToken() // consume comma
				p.nextToken() // move to next argument

				expr := p.parseExpression(LOWEST)
				if expr == nil {
					p.addError(fmt.Sprintf("expected expression after ',' at %s", p.curToken.Position))
					return nil
				}

				call.Arguments = append(call.Arguments, expr)
			}

			// Check for ORDER BY clause inside the function
			if p.peekTokenIsKeyword("ORDER") {
				p.nextToken() // consume ORDER

				// Expect BY keyword
				if !p.expectKeyword("BY") {
					return nil
				}

				p.nextToken() // move to first expression

				// Parse first ORDER BY expression
				orderByExpr := p.parseOrderByExpression()
				if orderByExpr.Expression != nil {
					call.OrderBy = append(call.OrderBy, orderByExpr)
				}

				// Parse additional ORDER BY expressions separated by commas
				for p.peekTokenIsPunctuator(",") {
					p.nextToken() // consume comma
					p.nextToken() // move to next ORDER BY expression

					orderByExpr := p.parseOrderByExpression()
					call.OrderBy = append(call.OrderBy, orderByExpr)
				}
			}

			// Check for ORDER BY clause inside the function
			if p.peekTokenIsKeyword("ORDER") {
				p.nextToken() // consume ORDER

				// Expect BY keyword
				if !p.expectKeyword("BY") {
					return nil
				}

				p.nextToken() // move to first expression

				// Parse first ORDER BY expression
				orderByExpr := p.parseOrderByExpression()
				if orderByExpr.Expression != nil {
					call.OrderBy = append(call.OrderBy, orderByExpr)
				}

				// Parse additional ORDER BY expressions separated by commas
				for p.peekTokenIsPunctuator(",") {
					p.nextToken() // consume comma
					p.nextToken() // move to next ORDER BY expression

					orderByExpr := p.parseOrderByExpression()
					call.OrderBy = append(call.OrderBy, orderByExpr)
				}
			}

			// Expect closing parenthesis
			if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
				p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
				return nil
			}

			return call
		}

		// Standard function call with arguments
		p.nextToken() // Move to the first argument

		// Parse the first argument
		firstArg := p.parseExpression(LOWEST)
		if firstArg != nil {
			call.Arguments = append(call.Arguments, firstArg)
		}

		// Parse additional arguments
		for p.peekTokenIsPunctuator(",") {
			p.nextToken() // consume comma
			p.nextToken() // move to next argument

			arg := p.parseExpression(LOWEST)
			if arg != nil {
				call.Arguments = append(call.Arguments, arg)
			} else {
				p.addError(fmt.Sprintf("expected expression after ',' at %s", p.curToken.Position))
				return nil
			}
		}

		// Check for ORDER BY clause inside the function
		if p.peekTokenIsKeyword("ORDER") {
			p.nextToken() // consume ORDER

			// Expect BY keyword
			if !p.expectKeyword("BY") {
				return nil
			}

			p.nextToken() // move to first expression

			// Parse first ORDER BY expression
			orderByExpr := p.parseOrderByExpression()
			call.OrderBy = append(call.OrderBy, orderByExpr)

			// Parse additional ORDER BY expressions separated by commas
			for p.peekTokenIsPunctuator(",") {
				p.nextToken() // consume comma
				p.nextToken() // move to next ORDER BY expression

				orderByExpr := p.parseOrderByExpression()
				call.OrderBy = append(call.OrderBy, orderByExpr)
			}
		}

		// Expect closing parenthesis
		if !p.expectPeek(TokenPunctuator) || p.curToken.Literal != ")" {
			p.addError(fmt.Sprintf("expected ')', got %s at %s", p.curToken.Literal, p.curToken.Position))
			return nil
		}

		return call
	case "*":
		// Multiplication operator - create infix expression
		expression := &InfixExpression{
			Token:    p.curToken,
			Operator: p.curToken.Literal,
			Left:     left,
		}

		// Get the precedence of the current operator
		precedence := PRODUCT // Explicitly use PRODUCT precedence for multiplication

		// Parse the right expression with the correct precedence
		p.nextToken()
		expression.Right = p.parseExpression(precedence)
		if expression.Right == nil {
			p.addError(fmt.Sprintf("expected expression after '*' at %s", p.curToken.Position))
			return nil
		}

		return expression
	default:
		p.addError(fmt.Sprintf("unexpected infix punctuator: %s at %s", p.curToken.Literal, p.curToken.Position))
		return nil
	}
}

// parseShowStatement parses a SHOW statement (SHOW TABLES, SHOW CREATE TABLE, SHOW INDEXES)
func (p *Parser) parseShowStatement() Statement {
	token := p.curToken

	// Check what kind of SHOW statement this is
	if p.peekTokenIsKeyword("TABLES") {
		p.nextToken() // consume TABLES
		return &ShowTablesStatement{
			Token: token,
		}
	} else if p.peekTokenIsKeyword("CREATE") {
		p.nextToken() // consume CREATE

		// Expect TABLE
		if !p.expectKeyword("TABLE") {
			return nil
		}

		// Expect table name
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}

		return &ShowCreateTableStatement{
			Token: token,
			TableName: &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			},
		}
	} else if p.peekTokenIsKeyword("INDEXES") || p.peekTokenIsKeyword("INDEX") {
		p.nextToken() // consume INDEXES/INDEX

		// Expect FROM
		if !p.expectKeyword("FROM") {
			return nil
		}

		// Expect table name
		if !p.expectPeek(TokenIdentifier) {
			return nil
		}

		return &ShowIndexesStatement{
			Token: token,
			TableName: &Identifier{
				Token: p.curToken,
				Value: p.curToken.Literal,
			},
		}
	}

	p.addError(fmt.Sprintf("unsupported SHOW statement at %s", p.curToken.Position))
	return nil
}

// Utility functions for checking the precedence of operators

// peekPrecedence returns the precedence of the next token
func (p *Parser) peekPrecedence() int {
	if p.peekToken.Type == TokenOperator {
		if prec, ok := precedences[p.peekToken.Literal]; ok {
			return prec
		}
	} else if p.peekToken.Type == TokenKeyword {
		if prec, ok := precedences[strings.ToUpper(p.peekToken.Literal)]; ok {
			return prec
		}
	} else if p.peekToken.Type == TokenPunctuator && p.peekToken.Literal == "." {
		return DOT
	} else if p.peekToken.Type == TokenPunctuator && p.peekToken.Literal == "(" {
		return CALL
	}
	return LOWEST
}

// curPrecedence returns the precedence of the current token
func (p *Parser) curPrecedence() int {
	if p.curToken.Type == TokenOperator {
		if prec, ok := precedences[p.curToken.Literal]; ok {
			return prec
		}
	} else if p.curToken.Type == TokenKeyword {
		if prec, ok := precedences[strings.ToUpper(p.curToken.Literal)]; ok {
			return prec
		}
	} else if p.curToken.Type == TokenPunctuator && p.curToken.Literal == "." {
		return DOT
	} else if p.curToken.Type == TokenPunctuator && p.curToken.Literal == "(" {
		return CALL
	}
	return LOWEST
}
